import sys
import types
import unittest
from unittest.mock import patch

from coding_agent import (
    LANGUAGES,
    _make_prompt,
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
    def test_language_detection_and_wide_catalog(self):
        self.assertEqual(detect_language("Сделай скрипт на Godot", "auto"), "gdscript")
        self.assertEqual(detect_language("Create a FastAPI endpoint", "auto"), "python")
        self.assertEqual(detect_language("write a Rust function", "auto"), "rust")
        self.assertEqual(detect_language("whatever", "kotlin"), "kotlin")
        self.assertEqual(detect_language("whatever", "C++"), "cpp")
        for language in ("python", "gdscript", "java", "javascript", "typescript",
                         "c", "cpp", "csharp", "go", "rust", "php", "ruby",
                         "swift", "dart", "sql", "bash", "powershell", "lua",
                         "r", "scala", "perl", "elixir", "haskell", "html",
                         "css", "xml", "json", "yaml", "toml", "dockerfile",
                         "glsl", "solidity", "other"):
            self.assertIn(language, LANGUAGES)

    def test_python_validation_uses_ast_without_executing(self):
        good = validate_code("def f(x):\n    return x + 1\n", "python")
        bad = validate_code("def f(:\n", "python")
        self.assertEqual(good["status"], "passed")
        self.assertEqual(good["method"], "python_ast")
        self.assertEqual(bad["status"], "failed")

    def test_strict_data_format_validation(self):
        self.assertEqual(validate_code('{"ok": true}', "json")["status"], "passed")
        self.assertEqual(validate_code('{"ok":', "json")["status"], "failed")
        self.assertEqual(validate_code("<root><item/></root>", "xml")["status"], "passed")
        self.assertEqual(validate_code("<root>", "xml")["status"], "failed")
        self.assertEqual(validate_code('name = "swin"\n', "toml")["status"], "passed")

    def test_prompt_preserves_task_and_fits_model_context(self):
        task = "Сделай большую программу. " * 50 + "ВАЖНО: верни полный код класса Player."
        prompt = _make_prompt(
            task,
            "python",
            [{"title": "Docs", "url": "https://docs.python.org/3/", "snippet": "Документация " * 100}],
            "Существующий код проекта " * 30,
            block_size=256,
        )
        self.assertLessEqual(len(prompt.encode("utf-8")), 252)
        self.assertIn("ЗАДАЧА:", prompt)
        self.assertIn("ВАЖНО:", prompt)
        self.assertIn("КОД:", prompt)

    def test_search_covers_general_internet_and_rejects_insecure_links(self):
        fake_module = types.SimpleNamespace(
            DDGS=lambda **kwargs: types.SimpleNamespace(
                text=lambda *args, **kw: [
                    {"title": "Community", "href": "https://stackoverflow.com/questions/1", "body": "Community example"},
                    {"title": "Docs", "href": "https://docs.python.org/3/", "body": "Python docs"},
                    {"title": "Unsafe", "href": "http://example.com", "body": "No TLS"},
                    {"title": "Local", "href": "https://127.0.0.1/private", "body": "Local"},
                    {"title": "Bad", "href": "javascript:alert(1)", "body": "Bad"},
                ]
            )
        )
        with patch.dict(sys.modules, {"ddgs": fake_module}):
            results = search_documentation("parse json", "python")
        urls = [item["url"] for item in results]
        self.assertIn("https://stackoverflow.com/questions/1", urls)
        self.assertIn("https://docs.python.org/3/", urls)
        self.assertTrue(all(url.startswith("https://") for url in urls))
        self.assertEqual(results[0]["trust"], "primary")

    def test_activity_is_hidden_by_default_and_verification_is_returned(self):
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
        self.assertEqual(result["verification"]["syntax"]["status"], "passed")
        self.assertEqual(result["verification"]["runtime_tests"]["status"], "not_run")

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
        self.assertIn("verification", result)


if __name__ == "__main__":
    unittest.main()
