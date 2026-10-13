import os
import json
import time
import random
from bs4 import BeautifulSoup
from ddgs import DDGS

try:
    from crawl4ai import AsyncWebCrawler
except Exception: 
    AsyncWebCrawler = None

FOOTBALL_KEYWORDS = [
    "football",
    "soccer",
    "player",
    "club",
    "winger",
    "striker",
    "midfielder",
    "defender",
    "goalkeaper",
    "goal"
]

TRUSTED_SOURCES = [
    "totalfootballanalysis.com", "breakingthelines.com", "theanalyst.com", 
    "coachesvoice.com", "theathletic.com", "fotmob.com", "sofascore.com",
    "espn.com", "espn.in", "bbc.com", "bbc.co.uk", "skysports.com", 
    "goal.com", "90min.com", "vavel.com", "tribuna.com", "beinsports.com",
    "gazzetta.it", "corrieredellosport.it", "tuttosport.com", "sky.it", 
    "sportmediaset.mediaset.it", "gianlucadimarzio.com", "alfredopedulla.com", 
    "calciomercato.com", "tuttomercatoweb.com", "tutto-m.com", "parlandodicalcio.it",
    "marca.com", "as.com", "mundodeportivo.com", "lequipe.fr", "footmercato.net",
    "kicker.de", "bild.de", "dailymail.co.uk", "theguardian.com", "globoesporte.globo.com"
]

BLOCKED_DOMAINS = [
    "facebook.com", 
    "twitter.com", 
    "instagram.com", 
    "linkedin.com", 
    "api-football.com"
]

MAX_ARTICLES = 3

class ArticleRetriever:

    def __init__(self, players_path="data/players.json"):
        self.players_path = players_path
        parent_dir = os.path.dirname(self.players_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def _search_query_with_retry(self, ddgs, query):
        for attempt in range(2):
            try:
                return list(ddgs.text(query, max_results=10, timeout=20))
            except Exception as e:
                if attempt < 2:
                    wait_time = 5
                    print(f"Errore query {query} (tentativo {attempt + 1}/2): {e}. Riprovo tra {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Errore query {query}: {e}")
        return []

    def search_articles(self, player_name):
        queries = [
            f'"{player_name}" scouting report',
            f'"{player_name}" tactical analysis',
            f'"{player_name}" player profile',
            f'"{player_name}" focus'
        ]

        all_results = []
        with DDGS(timeout=20) as ddgs:
            for query in queries:
                results = self._search_query_with_retry(ddgs, query)
                if results:
                    all_results.extend(results)
                    break

                all_results.extend(results)
                time.sleep(0.5)

        filtered = []
        seen_urls = set()
        added_articles = 0

        for result in all_results:
            url = result.get("href", "")
            if not url or url in seen_urls:
                continue

            if any(domain in url for domain in BLOCKED_DOMAINS):
                continue

            text = (result.get("title", "") + " " + result.get("body", "")).lower()
            player_tokens = [token for token in player_name.lower().split() if len(token) > 2]
            player_matches = any(token in text for token in player_tokens)

            if not player_matches:
                continue
            if not any(k in text for k in FOOTBALL_KEYWORDS):
                continue

            seen_urls.add(url)
            filtered.append(result)
            added_articles += 1
            if added_articles == 3:
                continue
        filtered.sort(key=lambda x: any(source in x.get("href", "") for source in TRUSTED_SOURCES), reverse=True)
        return filtered[:MAX_ARTICLES]

    def download_article(self, url):
        if AsyncWebCrawler is None:
            return ""

        try:
            from crawl4ai import AsyncWebCrawler as Crawl4AIWebCrawler

            async def _fetch():
                crawler = Crawl4AIWebCrawler(verbose=False)
                async with crawler as crawler_instance:
                    result = await crawler_instance.arun(url=url, delay_before_return_html=1.0)
                    if result and getattr(result, "markdown", None):
                        return result.markdown.strip()
                    if result and getattr(result, "html", None):
                        return BeautifulSoup(result.html, "html.parser").get_text(" ", strip=True).strip()
                return ""

            import asyncio
            return asyncio.run(_fetch())
        except Exception as e:
            print(f"Crawl4AI fallito per {url}: {e}")
            return ""

    def retrieve_player_articles(self, player):
        search_results = self.search_articles(
            player["name"]
        )
        if not search_results:
            return []

        articles = []
        for result in search_results:
            url = result.get("href")
            if not url:
                continue

            time.sleep(random.uniform(1.5, 3.5))
            text = self.download_article(url)

            if len(text) < 500:
                continue
         
            articles.append({
                "title": result.get("title", ""),
                "url": url,
                "player": player["name"],
                "text": text
            })
        return articles

    def add_article_to_player(self, player, article):
        if "articles" not in player:
            player["articles"] = []

        player["articles"].append({
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "text": article.get("text", "")
        })
    
    def write_json_players_file(self, team_id, player_list):
        current_dataset = []
        if os.path.exists(self.players_path):
            try:
                with open(self.players_path, "r", encoding="utf-8") as jsonFile:
                    current_dataset = json.load(jsonFile)
                    if not isinstance(current_dataset, list):
                        current_dataset = []
            except json.JSONDecodeError:
                print(f"⚠️ Il file {self.players_path} era corrotto o vuoto. Ne creo uno nuovo.")
                current_dataset = []

        filtered_dataset = [
            player for player in current_dataset
            if player.get("team_id") != team_id
        ]

        players_to_add = []
        for player in player_list or []:
            if isinstance(player, dict):
                player_copy = dict(player)
                player_copy["team_id"] = player_copy.get("team_id", team_id)
                players_to_add.append(player_copy)

        updated_dataset = filtered_dataset + players_to_add

        print(f"Rimossi {len(current_dataset) - len(filtered_dataset)} vecchi record per la squadra {team_id}")
        print(f"Aggiunti/Aggiornati {len(players_to_add)} giocatori per la squadra {team_id}")

        try:
            with open(self.players_path, "w", encoding="utf-8") as jsonFile:
                json.dump(updated_dataset, jsonFile, indent=2, ensure_ascii=False)
            print(f"File JSON aggiornato con successo per {team_id}!")
        except Exception as e:
            print(f"Errore nella scrittura del file JSON: {e}")

    def write_full_players_file(self, player_list):
        players_to_write = []
        for player in player_list or []:
            if isinstance(player, dict):
                player_copy = dict(player)
                players_to_write.append(player_copy)

        try:
            with open(self.players_path, "w", encoding="utf-8") as jsonFile:
                json.dump(players_to_write, jsonFile, indent=2, ensure_ascii=False)
            print(f"File JSON completo aggiornato con {len(players_to_write)} giocatori!")
        except Exception as e:
            print(f"Errore nella scrittura del file JSON completo: {e}")