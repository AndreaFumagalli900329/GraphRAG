import requests
import os
import time
import json

MAPPING_FILE = "data/team_id.json"

class FootballAPIClient:

    def __init__(self):
        self.base_url = "https://v3.football.api-sports.io"

        self.api_key = os.getenv("API_FOOTBALL_KEY")

        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY non trovata nel .env")

        self.headers = {
            "x-apisports-key": self.api_key
        }

    def clean_stat(self, value):
        if value is None:
            return 0
        return value

    def _get(self, endpoint, params=None):
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=20
            )

            # Rate limit handling base
            if response.status_code == 429:
                print("Rate limit raggiunto, attendo 5s...")
                time.sleep(5)
                return self._get(endpoint, params)

            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"Errore API ({endpoint}): {e}")
            return None

    def get_teams_by_league(self, league_id, season):
        return self._get(
            "teams",
            {
                "league": league_id,
                "season": season
            }
        )

    def get_players_by_team(self, team_id, season):
        return self._get(
            "players",
            {
                "team": team_id,
                "season": season
            }
        )
    
    def _get_all_players_pages(self, team_id, season):
        players = []
        page = 1

        while True:
            data = self._get(
                "players",
                {
                    "team": team_id,
                    "season": season,
                    "page": page
                }
            )
            if not data:
                break
            players.extend(data.get("response", []))
            paging = data.get("paging", {})
            if page >= paging.get("total", 1):
                break
            page += 1
            time.sleep(0.3)

        return players
    
    def _parse_player(self, item, team_id):
        player = item.get("player", {})
        stats_list = item.get("statistics", [])

        if not player:
            return None

        stats = stats_list[0] if stats_list else {}

        games = stats.get("games", {})
        goals = stats.get("goals", {})
        shots = stats.get("shots", {})
        passes = stats.get("passes", {})
        tackles = stats.get("tackles", {})
        dribbles = stats.get("dribbles", {})

        team_info = stats.get("team", {})

        return {
            "id": player.get("id"),
            "name": player.get("name"),
            "age": player.get("age"),
            "team": team_info.get("name"),
            "team_id": team_id,
            "position": games.get("position"),
            "nationality": player.get("nationality"),
            "statistics": {
                "appearances": self.clean_stat(games.get("appearances")),
                "minutes": self.clean_stat(games.get("minutes")),

                "goals": self.clean_stat(goals.get("total")),
                "assists": self.clean_stat(goals.get("assists")),

                "shots": self.clean_stat(shots.get("total")),
                "shots_on_target": self.clean_stat(shots.get("on")),

                "passes": self.clean_stat(passes.get("total")),
                "pass_accuracy": self.clean_stat(passes.get("accuracy")),

                "tackles": self.clean_stat(tackles.get("total")),
                "interceptions": self.clean_stat(tackles.get("interceptions")),

                "dribbles": self.clean_stat(dribbles.get("success")),
            }
        }

    def get_all_players_from_team(self, team_id, season):
        players_data = self._get_all_players_pages(team_id, season)
        players_list = []
        seen = set()
        for item in players_data:
            parsed = self._parse_player(item, team_id)
            if not parsed:
                continue
            if parsed["id"] in seen:
                continue
            seen.add(parsed["id"])
            players_list.append(parsed)
        return players_list

    def get_all_players_from_league(self, league_id, season):
        teams_data = self.get_teams_by_league(league_id, season)
        if not teams_data:
            return []
        players_list = []

        for team_entry in teams_data.get("response", []):
            team = team_entry.get("team", {})
            team_id = team.get("id")
            team_name = team.get("name")
            if not team_id:
                continue
            print(f"Recupero giocatori {team_name}...")
            players_list.extend(self.get_all_players_from_team(team_id, season))

        return players_list
    
    def get_team_id_by_name(self, team_name):
        if not os.path.exists(MAPPING_FILE):
            return None
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return mapping.get(team_name)

    def get_available_team_names(self):
        if not os.path.exists(MAPPING_FILE):
            return []
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return sorted(mapping.keys())