import json
import os
import tempfile
import unittest
from unittest.mock import patch

from football_api import FootballAPIClient


class TestFootballAPIClient(unittest.TestCase):
    def test_get_available_teams_reads_team_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_path = os.path.join(tmpdir, "team_id.json")
            with open(mapping_path, "w", encoding="utf-8") as handle:
                json.dump({"Lazio": 487, "Milan": 489}, handle)

            with patch.dict(os.environ, {"API_FOOTBALL_KEY": "test-key"}, clear=False):
                with patch("football_api.MAPPING_FILE", mapping_path):
                    client = FootballAPIClient()

                    self.assertEqual(client.get_available_team_names(), ["Lazio", "Milan"])
                    self.assertEqual(client.get_team_id_by_name("Lazio"), 487)

    def test_get_teams_by_league_reads_multiple_pages(self):
        with patch.dict(os.environ, {"API_FOOTBALL_KEY": "test-key"}, clear=False):
            client = FootballAPIClient()

            responses = [
                {
                    "response": [
                        {"team": {"id": 1, "name": "Lazio"}}
                    ],
                    "paging": {"current": 1, "total": 2},
                },
                {
                    "response": [
                        {"team": {"id": 2, "name": "Milan"}}
                    ],
                    "paging": {"current": 2, "total": 2},
                },
            ]

            with patch.object(client, "_get", side_effect=responses) as mocked_get:
                teams = client.get_teams_by_league(135, 2024)

            self.assertEqual(len(teams["response"]), 2)
            self.assertEqual([team["team"]["name"] for team in teams["response"]], ["Lazio", "Milan"])
            self.assertEqual(mocked_get.call_count, 2)

    def test_get_players_by_team_reads_multiple_pages(self):
        with patch.dict(os.environ, {"API_FOOTBALL_KEY": "test-key"}, clear=False):
            client = FootballAPIClient()

            responses = [
                {
                    "response": [
                        {
                            "player": {"id": 1, "name": "A"},
                            "statistics": [
                                {
                                    "team": {"name": "Lazio"},
                                    "games": {"position": "FW", "appearences": 1, "minutes": 90},
                                    "goals": {"total": 1, "assists": 0},
                                    "shots": {"total": 2, "on": 1},
                                    "passes": {"total": 10, "accuracy": 80},
                                    "tackles": {"total": 1, "interceptions": 0},
                                    "dribbles": {"success": 1},
                                }
                            ],
                        }
                    ],
                    "paging": {"current": 1, "total": 2},
                },
                {
                    "response": [
                        {
                            "player": {"id": 2, "name": "B"},
                            "statistics": [
                                {
                                    "team": {"name": "Lazio"},
                                    "games": {"position": "MF", "appearences": 2, "minutes": 180},
                                    "goals": {"total": 0, "assists": 1},
                                    "shots": {"total": 0, "on": 0},
                                    "passes": {"total": 20, "accuracy": 90},
                                    "tackles": {"total": 2, "interceptions": 1},
                                    "dribbles": {"success": 2},
                                }
                            ],
                        }
                    ],
                    "paging": {"current": 2, "total": 2},
                },
                {"response": []},
            ]

            with patch.object(client, "_get", side_effect=responses) as mocked_get:
                players = client.get_all_players_from_team(487, 2024)

            self.assertEqual(len(players), 2)
            self.assertEqual([player["name"] for player in players], ["A", "B"])
            self.assertEqual(mocked_get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
