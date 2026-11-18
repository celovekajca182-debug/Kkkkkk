import sys
import types
import unittest
from unittest.mock import patch

from coding_agent import (
    detect_language,
    run_code_assist,
    search_documentation,
    validate_code,
)


class FakeModel:
    block_size = 256

    def __init__(self, response="def add(a, b):\n    return a + b\n"):
        self.response = response
        self.last_prompt = ""

    def generate(self, prompt, **kwargs):
        self.last_prompt = prompt
        return self.response


class CodingAgentTests(unittest.TestCase):
    def test_language_detection(self):
        self.assertEqual(detect_language("Сделай скрипт на Godot", "auto"), "gdscript")
        self.assertEqual(detect_language("Create a FastAPI endpoint", "auto"), "python")
        self.assertEqual(detect_language("whatever", "kotlin"), "kotlin")

    def test_python_validation_uses_ast_without_executing(self):
        good = validate_code("def f(x):\n    return x + 1\n", "python")
        bad = validate_code("def f(:\n", "python")
        self.assertEqual(good["status"], "passed")
        self.assertEqual(good["method"], "python_ast")
        self.assertEqual(bad["status"], "failed")

    def test_search_filters_non_https_links(self):
        fake_module = types.SimpleNamespace(
            DDGS=lambda **kwargs: types.SimpleNamespace(
                text=lambda *args, **kw: [
                    {"title": "Docs", "href": "https://docs.python.org/3/", "body": "Python docs"},
                    {"title": "Unsafe", "href": "http://example.com", "body": "No TLS"},
                    {"title": "Bad", "href": "javascript:alert(1)", "body": "Bad"},
                ]
            )
        )
        with patch.dict(sys.modules, {"ddgs": fake_module}):
            results = search_documentation("parse json", "python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Docs")

    def test_activity_is_hidden_by_default(self):
        model = FakeModel()
        result = run_code_assist(
            "Напиши функцию сложения двух чисел",
            "python",
            model,
            research_web=False,
            show_activity=False,
        )
        self.assertEqual(result["status"], "draft_validated")
        self.assertIn("def add", result["code"])
        self.assertNotIn("activity", result)
        self.assertIn("ЗАДАЧА", model.last_prompt)

    def test_activity_can_be_shown_without_private_reasoning(self):
        model = FakeModel()
        result = run_code_assist(
            "Напиши функцию сложения",
            "python",
            model,
            research_web=False,
            show_activity=True,
        )
        self.assertIn("activity", result)
        joined = " ".join(item["label"] for item in result["activity"])
        self.assertIn("Черновик", joined)
        self.assertNotIn("chain of thought", joined.lower())

    def test_missing_model_does_not_fake_code(self):
        result = run_code_assist(
            "Напиши API",
            "python",
            None,
            research_web=False,
        )
        self.assertEqual(result["status"], "model_not_loaded")
        self.assertEqual(result["code"], "")


if __name__ == "__main__":
    unittest.main()
