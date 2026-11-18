import tempfile
import unittest
from pathlib import Path

from train_transformer import (
    build_instruction_corpus,
    build_pretraining_corpus,
    build_tensor_splits,
    read_instruction_file,
)


class TrainingCorpusTests(unittest.TestCase):
    def test_instruction_jsonl_is_formatted_as_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(
                '{"question":"Как начать?","answer":"Сначала проверь проект."}\n',
                encoding="utf-8",
            )
            records = read_instruction_file(path)
        self.assertEqual(records, ["Пользователь: Как начать?\nSWIN: Сначала проверь проект."])

    def test_pretraining_and_sft_corpora_are_separated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.txt").write_text("Открытый текстовый документ для обучения.", encoding="utf-8")
            (root / "web.txt").write_text("Дополнительный веб-документ на русском языке.", encoding="utf-8")
            (root / "knowledge.json").write_text(
                '[{"title":"Тест","question":"Что это?","answer":"Это проверенный ответ."}]',
                encoding="utf-8",
            )
            (root / "dialogue.jsonl").write_text(
                '{"question":"Как проверить?","answer":"Запусти тесты."}\n',
                encoding="utf-8",
            )
            pretrain = build_pretraining_corpus(root / "base.txt", root / "web.txt")
            sft = build_instruction_corpus(
                root / "knowledge.json", root / "dialogue.jsonl", root / "missing.jsonl"
            )
        self.assertIn("Дополнительный веб-документ", pretrain)
        self.assertNotIn("Что это?", pretrain)
        self.assertIn("Пользователь: Что это?", sft)
        self.assertNotIn("веб-документ", sft)

    def test_tensor_splits_use_distinct_validation_corpus(self):
        alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        train = alphabet * 40
        validation = "БВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯА" * 20
        train_data, validation_data = build_tensor_splits(train, validation, block_size=32)
        self.assertGreater(train_data.numel(), 900)
        self.assertGreater(validation_data.numel(), 400)
        self.assertEqual(train_data[:2].tolist(), [208, 144])
        self.assertEqual(validation_data[:2].tolist(), [208, 145])


if __name__ == "__main__":
    unittest.main()
