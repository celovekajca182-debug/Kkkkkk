import unittest

from prepare_coding_data import valid_example


class PrepareCodingDataTests(unittest.TestCase):
    def test_keeps_valid_python_task_and_tests(self):
        item = valid_example({
            "task_id": 601,
            "text": "Write a function that adds two integers.",
            "code": "def add(a, b):\n    return a + b\n",
            "test_list": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
        })
        self.assertIsNotNone(item)
        self.assertIn("Задача Python", item["question"])
        self.assertIn("Проверочные примеры", item["question"])
        self.assertIn("def add", item["answer"])

    def test_drops_syntactically_invalid_code(self):
        item = valid_example({
            "task_id": 602,
            "text": "Write a function.",
            "code": "def broken(:\n pass",
            "test_list": [],
        })
        self.assertIsNone(item)

    def test_drops_empty_or_oversized_examples(self):
        self.assertIsNone(valid_example({"text": "", "code": "pass"}))
        self.assertIsNone(valid_example({
            "text": "task",
            "code": "x = 1\n" * 2000,
        }))


if __name__ == "__main__":
    unittest.main()
