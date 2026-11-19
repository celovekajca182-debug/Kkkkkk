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

    def test_curated_dialogue_is_available_for_inference(self):
        result = self.base.search("Что такое токенизатор SWIN ByteTokenizer?")
        self.assertIsNotNone(result)
        self.assertIn("UTF-8", result["answer"])

    def test_finewiki_articles_are_indexed_for_rag(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge.json").write_text("[]", encoding="utf-8")
            (root / "finewiki_ru_train.txt").write_text(
                "Биология\nБиология изучает живые организмы, клетки, наследственность и эволюцию. "
                "Она исследует разнообразие жизни на Земле.\n\n"
                "<|document|>\n"
                "Клетка\nКлетка является структурной и функциональной единицей живых организмов. "
                "Мембрана регулирует обмен веществ между клеткой и средой.\n",
                encoding="utf-8",
            )
            base = KnowledgeBase(root / "knowledge.json")
            result = base.search_many("Что такое клетка и как работает мембрана?", limit=2)
            self.assertTrue(result)
            self.assertEqual(result[0]["title"], "Клетка")
            self.assertIn("мембрана", result[0]["snippet"].lower())
            self.assertIn("ru.wikipedia.org/wiki", result[0]["url"])
            self.assertIn("CC BY-SA 4.0", result[0]["source_type"])
            self.assertEqual(base.wiki_articles, 2)

    def test_knowledge_base_is_populated(self):
        self.assertGreaterEqual(self.base.size, 10)


if __name__ == "__main__":
    unittest.main()
