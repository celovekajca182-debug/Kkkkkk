import unittest
from pathlib import Path

from knowledge import KnowledgeBase


class KnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = Path(__file__).resolve().parents[1] / "server" / "data" / "knowledge.json"
        cls.base = KnowledgeBase(data_path)

    def test_godot_movement_lookup(self):
        result = self.base.search("Как сделать движение персонажа в Godot 4?")
        self.assertIsNotNone(result)
        self.assertIn("CharacterBody3D", result["answer"])

    def test_gradle_properties_lookup(self):
        result = self.base.search("Gradle не запускается из-за android.useAndroidX")
        self.assertIsNotNone(result)
        self.assertIn("gradle.properties", result["answer"])

    def test_unrelated_query_is_not_fabricated(self):
        result = self.base.search("погода на Марсе завтра")
        self.assertIsNone(result)

    def test_knowledge_base_is_populated(self):
        self.assertGreaterEqual(self.base.size, 10)


if __name__ == "__main__":
    unittest.main()
