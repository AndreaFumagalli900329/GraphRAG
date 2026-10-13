import json
import os
import tempfile
import unittest
from unittest.mock import patch

from article_retriever import ArticleRetriever


class TestArticleRetriever(unittest.TestCase):
    def test_add_article_to_player_stores_article_in_player_articles(self):
        retriever = ArticleRetriever()
        player = {"name": "Lionel Messi"}
        article = {"player": "Lionel Messi", "text": "Some article text"}

        retriever.add_article_to_player(player, article)

        self.assertEqual(player["articles"][0]["text"], "Some article text")

    @patch("article_retriever.DDGS")
    def test_search_articles_retries_after_timeout(self, ddgs_cls):
        retriever = ArticleRetriever()
        fake_ddgs = ddgs_cls.return_value.__enter__.return_value
        fake_ddgs.text.side_effect = [
            TimeoutError("timed out"),
            [{"title": "Lionel Messi scouting report", "body": "football player analysis", "href": "https://example.com/messi"}],
            [],
            [],
            [],
        ]

        results = retriever.search_articles("Lionel Messi")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["href"], "https://example.com/messi")
        self.assertGreaterEqual(fake_ddgs.text.call_count, 2)

    @patch("article_retriever.DDGS")
    def test_search_articles_stops_after_first_good_result(self, ddgs_cls):
        retriever = ArticleRetriever()
        fake_ddgs = ddgs_cls.return_value.__enter__.return_value
        fake_ddgs.text.return_value = [
            {"title": "Lionel Messi scouting report", "body": "football player analysis", "href": "https://example.com/messi"}
        ]

        results = retriever.search_articles("Lionel Messi")

        self.assertEqual(len(results), 1)
        self.assertEqual(fake_ddgs.text.call_count, 1)

    @patch("article_retriever.DDGS")
    def test_search_articles_accepts_surname_match(self, ddgs_cls):
        retriever = ArticleRetriever()
        fake_ddgs = ddgs_cls.return_value.__enter__.return_value
        fake_ddgs.text.return_value = [
            {"title": "Messi scouting report", "body": "football player analysis", "href": "https://example.com/messi"}
        ]

        results = retriever.search_articles("Lionel Messi")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["href"], "https://example.com/messi")

    @patch("article_retriever.DDGS")
    def test_search_articles_uses_longer_timeout_for_ddgs(self, ddgs_cls):
        retriever = ArticleRetriever()
        fake_ddgs = ddgs_cls.return_value.__enter__.return_value
        fake_ddgs.text.return_value = []

        retriever.search_articles("Lionel Messi")

        ddgs_cls.assert_called_once_with(timeout=20)

    def test_write_json_players_file_replaces_players_for_team(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            players_path = os.path.join(tmpdir, "players.json")
            initial_players = [
                {"id": 1, "name": "Old Player", "team_id": 487},
                {"id": 2, "name": "Other Team Player", "team_id": 488},
            ]
            with open(players_path, "w", encoding="utf-8") as handle:
                json.dump(initial_players, handle)

            retriever = ArticleRetriever(players_path=players_path)
            new_players = [{"id": 3, "name": "New Player", "team_id": 487}]

            retriever.write_json_players_file(487, new_players)

            with open(players_path, "r", encoding="utf-8") as handle:
                updated_players = json.load(handle)

            self.assertEqual([player["name"] for player in updated_players], ["Other Team Player", "New Player"])
            self.assertEqual([player["team_id"] for player in updated_players], [488, 487])

    def test_write_full_players_file_replaces_entire_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            players_path = os.path.join(tmpdir, "players.json")
            initial_players = [
                {"id": 1, "name": "Old Player", "team_id": 487},
            ]
            with open(players_path, "w", encoding="utf-8") as handle:
                json.dump(initial_players, handle)

            retriever = ArticleRetriever(players_path=players_path)
            new_players = [
                {"id": 2, "name": "New Player 1", "team_id": 488},
                {"id": 3, "name": "New Player 2", "team_id": 489},
            ]

            retriever.write_full_players_file(new_players)

            with open(players_path, "r", encoding="utf-8") as handle:
                updated_players = json.load(handle)

            self.assertEqual([player["name"] for player in updated_players], ["New Player 1", "New Player 2"])
            self.assertEqual(len(updated_players), 2)


if __name__ == "__main__":
    unittest.main()
