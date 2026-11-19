import tempfile
import unittest

import torch
from pathlib import Path

from transformer import BPEByteTokenizer, CharTokenizer

from train_transformer import (
    build_instruction_corpus,
    build_instruction_examples,
    build_pretraining_corpus,
    build_sft_windows,
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
            (root / "wiki.txt").write_text("Статья Википедии о космосе и науке.", encoding="utf-8")
            (root / "knowledge.json").write_text(
                '[{"title":"Тест","question":"Что это?","answer":"Это проверенный ответ."}]',
                encoding="utf-8",
            )
            (root / "dialogue.jsonl").write_text(
                '{"question":"Как проверить?","answer":"Запусти тесты."}\n',
                encoding="utf-8",
            )
            pretrain = build_pretraining_corpus(
                root / "base.txt",
                root / "web.txt",
                root / "wiki.txt",
            )
            sft = build_instruction_corpus(
                root / "knowledge.json", root / "dialogue.jsonl", root / "missing.jsonl"
            )
        self.assertIn("Дополнительный веб-документ", pretrain)
        self.assertIn("Статья Википедии", pretrain)
        self.assertNotIn("Что это?", pretrain)
        self.assertIn("Пользователь: Что это?", sft)
        self.assertNotIn("веб-документ", sft)

    def test_project_code_examples_are_included_in_instruction_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge.json").write_text("[]", encoding="utf-8")
            (root / "dialogue.jsonl").write_text("", encoding="utf-8")
            (root / "oasst.jsonl").write_text("", encoding="utf-8")
            (root / "mbpp.jsonl").write_text(
                '{"question":"Задача Python: сложить два числа","answer":"def add(a, b):\\n    return a + b"}\n',
                encoding="utf-8",
            )
            (root / "project.jsonl").write_text(
                '{"question":"Как двигать CharacterBody2D?","answer":"extends CharacterBody2D"}\n',
                encoding="utf-8",
            )
            sft = build_instruction_corpus(
                root / "knowledge.json",
                root / "dialogue.jsonl",
                root / "oasst.jsonl",
                root / "mbpp.jsonl",
                root / "project.jsonl",
            )
        self.assertIn("def add", sft)
        self.assertIn("CharacterBody2D", sft)

    def test_sft_windows_mask_user_prompt_and_train_only_assistant_answer(self):
        prompt = "Пользователь: Что такое функция в Python?"
        answer = "Функция — это именованный блок кода, который выполняет задачу и может вернуть результат."
        tokenizer = CharTokenizer.from_texts([prompt, answer, prompt + "\nSWIN:"])
        windows = build_sft_windows([(prompt, answer)], tokenizer, block_size=96)
        self.assertTrue(windows)

        input_ids, targets = windows[0]
        prompt_token_count = len(tokenizer.encode(prompt + "\nSWIN:"))
        valid_positions = [index for index, token in enumerate(targets.tolist()) if token != -100]
        self.assertEqual(valid_positions[0], prompt_token_count - 1)
        self.assertEqual(input_ids.numel(), 96)
        self.assertEqual(targets.numel(), 96)
        self.assertTrue(all(token == -100 for token in targets[:prompt_token_count - 1].tolist()))

    def test_sft_windows_cover_long_answers_with_overlapping_context(self):
        prompt = "Пользователь: Объясни обработку исключений в Python."
        answer = ("Используй try и except для обработки ожидаемых ошибок. " * 12)
        tokenizer = CharTokenizer.from_texts([prompt, answer, prompt + "\nSWIN:"])
        windows = build_sft_windows([(prompt, answer)], tokenizer, block_size=64)
        self.assertGreaterEqual(len(windows), 3)
        self.assertTrue(all(torch.any(targets != -100) for _, targets in windows))

    def test_sft_bpe_windows_include_eos_and_mask_prompt_tokens(self):
        prompt = "Пользователь: Напиши функцию Python."
        answer = "def add(a, b):\n    return a + b"
        corpus = (prompt + "\nSWIN: " + answer + "\n") * 80
        tokenizer = BPEByteTokenizer.from_texts([corpus], vocab_size=512)
        windows = build_sft_windows([(prompt, answer)], tokenizer, block_size=64)
        self.assertTrue(windows)
        self.assertTrue(any(torch.any(targets == tokenizer.eos_token_id) for _, targets in windows))
        _, targets = windows[0]
        prompt_count = len(tokenizer.encode(prompt + "\nSWIN:"))
        self.assertTrue(torch.all(targets[:max(0, prompt_count - 1)] == -100))

    def test_instruction_examples_are_deduplicated_and_keep_prompt_answer_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge.json").write_text(
                '[{"title":"Python","question":"Что такое функция?","answer":"Именованный блок кода."}]',
                encoding="utf-8",
            )
            (root / "dialogue.jsonl").write_text(
                '{"question":"Что такое функция?","answer":"Именованный блок кода."}\n',
                encoding="utf-8",
            )
            examples = build_instruction_examples(
                root / "knowledge.json",
                root / "dialogue.jsonl",
                root / "missing.jsonl",
            )
        self.assertEqual(len(examples), 1)
        self.assertIn("Пользователь:", examples[0][0])
        self.assertEqual(examples[0][1], "Именованный блок кода.")

    def test_tensor_splits_use_distinct_validation_corpus(self):
        alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        train = alphabet * 40
        validation = "БВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯА" * 20
        tokenizer = CharTokenizer.from_texts([train, validation])
        train_data, validation_data = build_tensor_splits(
            train, validation, block_size=32, tokenizer=tokenizer
        )
        self.assertGreater(train_data.numel(), 900)
        self.assertGreater(validation_data.numel(), 400)
        self.assertEqual(tokenizer.decode(train_data[:2].tolist()), train[:2])
        self.assertEqual(tokenizer.decode(validation_data[:2].tolist()), validation[:2])


if __name__ == "__main__":
    unittest.main()
