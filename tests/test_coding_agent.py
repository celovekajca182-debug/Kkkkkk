import sys
import types
import unittest
from unittest.mock import patch

from coding_agent import (
    LANGUAGES,
    _make_prompt,
    detect_language,
    looks_like_garbage_code,
    run_code_assist,
    retrieve_coding_examples,
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
            response_language="ru",
        )
        self.assertLessEqual(len(prompt.encode("utf-8")), 252)
        self.assertIn("User:", prompt)
        self.assertIn("ВАЖНО:", prompt)
        self.assertIn("Assistant:", prompt)

    def test_unicode_tokenizer_uses_character_context_instead_of_utf8_bytes(self):
        class CharacterTokenizer:
            vocabulary = ["<UNK>", "а", "б"]

            @staticmethod
            def encode(value):
                return list(value)

        task = "Создай функцию загрузки настроек и обработки ошибок."
        prompt = _make_prompt(
            task,
            "python",
            [],
            "Проект: Android-клиент SWIN AI, Python 3.11",
            block_size=256,
            tokenizer=CharacterTokenizer(),
            response_language="ru",
        )
        self.assertLessEqual(len(prompt), 252)
        self.assertIn(task, prompt)
        self.assertIn("Project context:", prompt)

    def test_bpe_tokenizer_uses_subword_count_for_context_budget(self):
        class PairBPETokenizer:
            @staticmethod
            def encode(value):
                return [value[index:index + 2] for index in range(0, len(value), 2)]

        task = "Создай функцию для загрузки настроек, проверки ошибок и сохранения конфигурации."
        prompt = _make_prompt(
            task,
            "python",
            [],
            "",
            block_size=256,
            tokenizer=PairBPETokenizer(),
            response_language="ru",
        )
        self.assertIn(task, prompt)
        self.assertTrue(prompt.startswith("Response language: Russian."))
        self.assertLessEqual(len(PairBPETokenizer.encode(prompt)), 252)

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

    def test_corrupt_repeated_generation_is_rejected(self):
        self.assertTrue(looks_like_garbage_code("оооооооооооооооооооо"))
        self.assertTrue(looks_like_garbage_code("код � сломан"))
        self.assertFalse(looks_like_garbage_code(
            "def add(a, b):\\n    return a + b"
        ))

    def test_corrupt_model_output_is_not_returned_as_code(self):
        model = FakeModel("оооооооооооооооооооо")
        result = run_code_assist(
            "Создай сложную Python функцию для обработки данных",
            "python",
            model,
            research_web=False,
        )
        self.assertEqual(result["status"], "generation_rejected")
        self.assertEqual(result["code"], "")
        self.assertEqual(result["validation"]["method"], "generation_quality_guard")

    def test_retrieves_relevant_godot_code_example(self):
        matches = retrieve_coding_examples(
            "Как создать объект или пулю из PackedScene в Godot 4?",
            "gdscript",
        )
        self.assertTrue(matches)
        self.assertIn("PackedScene", matches[0]["question"])
        self.assertGreaterEqual(matches[0]["score"], 0.68)

    def test_close_curated_example_is_used_without_free_form_generation(self):
        result = run_code_assist(
            "Как создать объект или пулю из PackedScene в Godot 4?",
            "gdscript",
            None,
            research_web=False,
        )
        self.assertEqual(result["status"], "retrieved_example")
        self.assertIn("instantiate()", result["code"])
        self.assertEqual(result["verification"]["syntax"]["status"], "limited")
        self.assertIn("matched_examples", result)

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
        self.assertIn("User:", model.last_prompt)
        self.assertEqual(result["response_language"], "ru")
        self.assertEqual(result["verification"]["syntax"]["status"], "passed")
        self.assertEqual(result["verification"]["runtime_tests"]["status"], "not_run")

    def test_code_assist_localizes_results_to_explicit_response_language(self):
        result = run_code_assist(
            "Write a Python function that adds two numbers.",
            "python",
            FakeModel(),
            research_web=False,
            response_language="de",
        )
        self.assertEqual(result["response_language"], "de")
        self.assertTrue(result["summary"].startswith("Ein Entwurf"))
        self.assertEqual(len(result["plan"]), 3)
        self.assertTrue(result["warnings"])
        self.assertTrue(all("Static validation" not in item for item in result["warnings"]))

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

    def test_cloud_text_generator_can_create_validated_code_without_local_checkpoint(self):
        calls = []

        def generator(prompt, max_tokens):
            calls.append((prompt, max_tokens))
            return "def format_display_name(name: str) -> str:\n    return name.strip().title()\n"

        result = run_code_assist(
            "Создай функцию Python для форматирования отображаемого имени пользователя",
            "python",
            None,
            research_web=False,
            text_generator=generator,
        )
        self.assertEqual(result["status"], "draft_validated")
        self.assertIn("format_display_name", result["code"])
        self.assertEqual(result["validation"]["status"], "passed")
        self.assertEqual(len(calls), 1)
        self.assertGreater(calls[0][1], 0)
        self.assertIn("User:", calls[0][0])

    def test_cloud_provider_failure_falls_back_to_local_swin(self):
        model = FakeModel("def local_answer(value):\n    return value.strip()\n")

        def unavailable_provider(prompt, max_tokens):
            raise RuntimeError("provider unavailable")

        result = run_code_assist(
            "Создай функцию Python для очистки строки",
            "python",
            model,
            research_web=False,
            text_generator=unavailable_provider,
            show_activity=True,
        )
        self.assertEqual(result["status"], "draft_validated")
        self.assertEqual(result["generation_engine"], "local_swin_transformer")
        self.assertIn("local_answer", result["code"])
        self.assertTrue(any(item["stage"] == "provider_fallback" for item in result["activity"]))

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
