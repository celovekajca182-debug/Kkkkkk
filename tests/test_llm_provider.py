import json
import os
import unittest
from unittest.mock import patch

from llm_provider import LLMProviderError, generate_text, llm_enabled


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return json.dumps(self.payload).encode("utf-8")


class LLMProviderTests(unittest.TestCase):
    def test_provider_requires_explicit_server_key(self):
        with patch.dict(os.environ, {"SWIN_LLM_API_KEY": ""}):
            self.assertFalse(llm_enabled())
            with self.assertRaises(LLMProviderError):
                generate_text("Тест", instructions="Ты SWIN.")

    def test_responses_api_request_uses_private_non_stored_request(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"output_text": "Готовый ответ SWIN."})

        with patch.dict(os.environ, {
            "SWIN_LLM_API_KEY": "test-secret-key",
            "SWIN_REASONING_MODEL": "gpt-6.2-sol",
        }):
            with patch("llm_provider.urllib.request.urlopen", side_effect=fake_urlopen):
                result = generate_text(
                    "Вопрос пользователя",
                    instructions="Отвечай точно.",
                    max_output_tokens=200,
                )

        self.assertEqual(result, "Готовый ответ SWIN.")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(captured["payload"]["input"][0]["role"], "developer")
        self.assertEqual(captured["payload"]["input"][0]["content"], "Отвечай точно.")
        self.assertEqual(captured["payload"]["input"][1]["role"], "user")
        self.assertEqual(captured["payload"]["input"][1]["content"], "Вопрос пользователя")
        self.assertNotIn("instructions", captured["payload"])
        self.assertEqual(captured["payload"]["model"], "gpt-6.2-sol")
        self.assertFalse(captured["payload"]["store"])
        self.assertEqual(captured["payload"]["max_output_tokens"], 200)
        self.assertIn("Bearer test-secret-key", captured["headers"]["Authorization"])

    def test_per_task_model_override_and_effort_are_applied(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse({"output_text": "Проверенный код."})

        with patch.dict(os.environ, {"SWIN_LLM_API_KEY": "test-key"}):
            with patch("llm_provider.urllib.request.urlopen", side_effect=fake_urlopen):
                generate_text(
                    "Сложная задача по коду",
                    instructions="Ты старший инженер.",
                    model_override="gpt-6.2-astra",
                    reasoning_effort="high",
                )
        self.assertEqual(captured["payload"]["model"], "gpt-6.2-astra")
        self.assertEqual(captured["payload"]["reasoning"]["effort"], "high")

    def test_invalid_empty_prompt_is_rejected(self):
        with patch.dict(os.environ, {"SWIN_LLM_API_KEY": "test-key"}):
            with self.assertRaisesRegex(LLMProviderError, "empty"):
                generate_text(" ", instructions="Answer.")


if __name__ == "__main__":
    unittest.main()
