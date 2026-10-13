import os
import json
import sys
from dotenv import load_dotenv, find_dotenv
from football_api import FootballAPIClient
from neo4j_loader import Neo4jLoader
from article_retriever import ArticleRetriever
from langchain_ollama import ChatOllama
from ui import FootballGraphUI
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

TEAM_ID = 489
LEAGUE_ID = 135

def load_environment_variables():
    load_dotenv(find_dotenv())
    
def load_players(path="data/players.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def connect_to_ollama():
    try:
        llm = ChatOllama(
            model="llama3.1",
            temperature=0,
            timeout=None
        )
        return llm
    except Exception as e:
        print(f"❌ ERRORE connessione Ollama: {e}")
        return None

def retrieve_from_team_api(api_client, retriever_client, season, team_id=None):
    print(f"Esecuzione chiamata API per la stagione {season}...")
    players_list = api_client.get_all_players_from_team(team_id, season)

    for player in players_list:
        articles = retriever_client.retrieve_player_articles(player)
        for article in articles:
            retriever_client.add_article_to_player(player, article)
    retriever_client.write_json_players_file(team_id, players_list)

    return players_list

def retrieve_from_league_api(api_client, retriever_client, season, league_id=None):
    print(f"Esecuzione chiamata API per la stagione {season}...") 
    players_list = api_client.get_all_players_from_league(league_id or LEAGUE_ID, season)

    for player in players_list:
        articles = retriever_client.retrieve_player_articles(player)
        for article in articles:
            retriever_client.add_article_to_player(player, article)
    retriever_client.write_full_players_file(players_list)

    return players_list

async def pipeline(neo4j_loader, llm):
    print("Avvio rigenerazione totale del grafo...")
    neo4j_loader.clear_graph()
    players = load_players()
    await neo4j_loader.build_graph(players, llm)

if __name__ == "__main__":
    load_environment_variables()

    llm = connect_to_ollama()
    api = FootballAPIClient()
    article_retriever = ArticleRetriever()

    neo4j_loader = Neo4jLoader(
        uri=os.environ.get("NEO4J_URI"),
        username=os.environ.get("NEO4J_USERNAME"),
        password=os.environ.get("NEO4J_PASSWORD")
    )

    app_ui = FootballGraphUI(
        neo4j_loader=neo4j_loader,
        llm=llm,
        api_client=api,
        article_retriever=article_retriever,
        pipeline_func=pipeline,
        retrieve_from_team_api_func=retrieve_from_team_api,
        retrieve_from_league_api_func=retrieve_from_league_api
    )
    
    app_ui.render()