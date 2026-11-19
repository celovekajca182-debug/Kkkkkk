import unittest
import types
from unittest.mock import patch

from prepare_corpus import (
    clean_text,
    extract_oasst_pairs,
    load_finewiki_documents,
    load_classic_documents,
    labels_have_rejection,
    normalize_oasst_language,
    russian_letter_ratio,
    script_letter_ratio,
    split_items,
)


def row(message_id, parent_id, role, text, **extra):
    value = {
        "message_id": message_id,
        "parent_id": parent_id,
        "role": role,
        "text": text,
        "lang": "ru",
        "review_result": True,
        "deleted": False,
        "synthetic": False,
        "labels": None,
    }
    value.update(extra)
    return value


class CorpusPreparationTests(unittest.TestCase):
    def test_masks_common_identifiers(self):
        text = clean_text("Почта user@example.com, IP 192.168.1.10, телефон +7 999 123-45-67")
        self.assertNotIn("user@example.com", text)
        self.assertNotIn("192.168.1.10", text)
        self.assertNotIn("+7 999 123-45-67", text)
        self.assertIn("[EMAIL]", text)
        self.assertIn("[IP]", text)
        self.assertIn("[PHONE]", text)

    def test_russian_letter_ratio(self):
        self.assertGreater(russian_letter_ratio("Это русский текст"), 0.95)
        self.assertLess(russian_letter_ratio("Mostly English text"), 0.1)

    def test_rejects_oasst_pii_label(self):
        # Actual Hugging Face OASST1 parquet schema: parallel arrays.
        labels = {
            "name": ["quality", "pii"],
            "value": [0.9, 0.25],
            "count": [3, 3],
        }
        self.assertTrue(labels_have_rejection(labels))
        clean_labels = {
            "name": ["quality"],
            "value": [0.9],
            "count": [3],
        }
        self.assertFalse(labels_have_rejection(clean_labels))
        # Also accept exports that map label names to value objects.
        self.assertTrue(
            labels_have_rejection({"quality": {"value": 0.9}, "pii": {"value": 0.25}})
        )

    def test_pairs_reviewed_russian_prompt_and_answer(self):
        rows = [
            row("q1", None, "prompter", "Как сделать движение в Godot 4?"),
            row("a1", "q1", "assistant", "Используй CharacterBody2D и move_and_slide()."),
            row("q2", None, "prompter", "Какой ответ нужно удалить?"),
            row(
                "a2",
                "q2",
                "assistant",
                "Плохой ответ с персональными данными.",
                labels={"name": ["pii"], "value": [0.5], "count": [2]},
            ),
            row("q3", None, "prompter", "How do I use Python?"),
            row("a3", "q3", "assistant", "Используй Python для скриптов."),
        ]
        pairs = extract_oasst_pairs(rows, max_pairs=10, seed=1)
        self.assertEqual(len(pairs), 1)
        self.assertIn("Godot", pairs[0]["question"])
        self.assertIn("CharacterBody2D", pairs[0]["answer"])

    def test_oasst_pairs_retain_language_and_balance_available_languages(self):
        rows = [
            row("q_ru", None, "prompter", "Как создать функцию в Python?", lang="ru"),
            row("a_ru", "q_ru", "assistant", "Функция принимает данные и возвращает результат.", lang="ru"),
            row("q_en", None, "prompter", "How do I create a Python function?", lang="en"),
            row("a_en", "q_en", "assistant", "A function receives values and returns a result.", lang="en"),
            row("q_uk", None, "prompter", "Як створити функцію у Python?", lang="uk"),
            row("a_uk", "q_uk", "assistant", "Функція приймає значення та повертає результат.", lang="uk"),
            row("q_pt", None, "prompter", "Como criar uma função em Python?", lang="pt-BR"),
            row("a_pt", "q_pt", "assistant", "Uma função recebe valores e retorna um resultado.", lang="pt-BR"),
            row("q_it", None, "prompter", "Come creare una funzione Python?", lang="it"),
            row("a_it", "q_it", "assistant", "Una funzione riceve valori e restituisce un risultato.", lang="it"),
            row("q_ja", None, "prompter", "Pythonで関数を作成するにはどうすればよいですか？", lang="ja"),
            row("a_ja", "q_ja", "assistant", "関数は値を受け取り、処理して結果を返します。", lang="ja"),
        ]
        pairs = extract_oasst_pairs(rows, max_pairs=20, seed=7)
        language_codes = {pair["language"] for pair in pairs}
        self.assertEqual(language_codes, {"ru", "en", "uk", "pt", "it", "ja"})
        self.assertTrue(all(pair["language"] for pair in pairs))

    def test_language_codes_normalize_regional_tags(self):
        self.assertEqual(normalize_oasst_language("pt-BR"), "pt")
        self.assertEqual(normalize_oasst_language("zh"), "zh")
        self.assertEqual(normalize_oasst_language("ja"), "ja")
        self.assertEqual(normalize_oasst_language("unknown"), "")

    def test_japanese_script_ratio_counts_kana_and_kanji(self):
        self.assertGreater(
            script_letter_ratio("日本語の文章です。ひらがなとカタカナも含みます。", "ja"),
            0.8,
        )
        self.assertLess(script_letter_ratio("Mostly English documentation", "ja"), 0.1)

    def test_finewiki_stream_keeps_russian_articles_and_splits_validation(self):
        class FakeStream(list):
            def shuffle(self, seed, buffer_size):
                return self

        rows = [
            {
                "title": f"Статья {index}",
                "text": ("Это подробная русская энциклопедическая статья о науке, истории и природе. " * 3),
                "url": f"https://ru.wikipedia.org/wiki/Статья_{index}",
            }
            for index in range(20)
        ]
        fake_datasets = types.SimpleNamespace(
            load_dataset=lambda *args, **kwargs: FakeStream(rows)
        )
        with patch.dict("sys.modules", {"datasets": fake_datasets}):
            train, validation, total_chars = load_finewiki_documents(
                max_documents=10,
                max_chars=10000,
                max_doc_chars=1000,
                min_doc_chars=30,
                validation_fraction=0.2,
                seed=17,
            )
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertLessEqual(len(train) + len(validation), 10)
        self.assertGreater(total_chars, 0)
        self.assertIn("Статья", train[0])

    def test_classic_stream_keeps_russian_text_and_splits_validation(self):
        class FakeStream(list):
            def shuffle(self, seed, buffer_size):
                return self

        rows = [
            {"text": f"Русский классический текст номер {index}. " + ("Описание природы и человеческих поступков. " * 10)}
            for index in range(20)
        ]
        fake_datasets = types.SimpleNamespace(
            load_dataset=lambda *args, **kwargs: FakeStream(rows)
        )
        with patch.dict("sys.modules", {"datasets": fake_datasets}):
            train, validation, total_chars = load_classic_documents(
                max_documents=10,
                max_chars=10000,
                max_doc_chars=1000,
                min_doc_chars=30,
                validation_fraction=0.2,
                seed=19,
            )
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertLessEqual(len(train) + len(validation), 10)
        self.assertGreater(total_chars, 0)
        self.assertIn("Русский", train[0])

    def test_split_is_reproducible_and_nonempty(self):
        items = list(range(20))
        train_a, valid_a = split_items(items, 0.1, 17)
        train_b, valid_b = split_items(items, 0.1, 17)
        self.assertEqual(train_a, train_b)
        self.assertEqual(valid_a, valid_b)
        self.assertTrue(train_a)
        self.assertTrue(valid_a)
        self.assertFalse(set(train_a) & set(valid_a))


if __name__ == "__main__":
    unittest.main()
