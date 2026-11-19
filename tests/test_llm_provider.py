import json
import unittest
from unittest.mock import patch

import llm_provider


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.payload


class LlmProviderTests(unittest.TestCase):
    def test_provider_is_disabled_without_base_url(self):
        with patch.object(llm_provider, "BASE_URL", ""):
            self.assertFalse(llm_provider.provider_configured())

    def test_model_roles_are_separate(self):
        self.assertEqual(
            llm_provider.model_for_role("chat"),
            "Qwen/Qwen3-4B-Instruct-2507",
        )
        self.assertEqual(
            llm_provider.model_for_role("code"),
            "Qwen/Qwen2.5-Coder-3B-Instruct",
        )

    def test_completion_uses_correct_model_and_auth(self):
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "print(2 + 2)"}}]
        }
        with (
            patch.object(llm_provider, "BASE_URL", "https://inference.example/v1"),
            patch.object(llm_provider, "API_KEY", "test-secret"),
            patch.object(llm_provider, "CODE_MODEL", "example/coder"),
            patch(
                "llm_provider.urllib.request.urlopen",
                return_value=FakeResponse(payload),
            ) as urlopen,
        ):
            result = llm_provider.generate_model_text(
                "Write Python", role="code", max_tokens=500, temperature=0.1
            )

        self.assertEqual(result, "print(2 + 2)")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://inference.example/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "example/coder")
        self.assertEqual(body["max_tokens"], 500)

    def test_full_completion_url_is_not_duplicated(self):
        with patch.object(llm_provider, "BASE_URL", "https://inference.example/v1/chat/completions"):
            with patch(
                "llm_provider.urllib.request.urlopen",
                return_value=FakeResponse({
                    "choices": [{"message": {"content": "ready"}}]
                }),
            ) as urlopen:
                self.assertEqual(
                    llm_provider.generate_model_text("ping", role="chat"),
                    "ready",
                )
        self.assertEqual(
            urlopen.call_args.args[0].full_url,
            "https://inference.example/v1/chat/completions",
        )

    def test_empty_prompt_is_rejected(self):
        with patch.object(llm_provider, "BASE_URL", "https://inference.example/v1"):
            with self.assertRaises(ValueError):
                llm_provider.generate_model_text("  ", role="chat")

    def test_list_content_blocks_are_supported(self):
        with patch.object(llm_provider, "BASE_URL", "https://inference.example/v1"):
            with patch(
                "llm_provider.urllib.request.urlopen",
                return_value=FakeResponse({
                    "choices": [{
                        "message": {
                            "content": [
                                {"type": "text", "text": "First "},
                                {"type": "text", "text": "second"},
                            ]
                        }
                    }]
                }),
            ):
                self.assertEqual(
                    llm_provider.generate_model_text("ping", role="chat"),
                    "First second",
                )


if __name__ == "__main__":
    unittest.main()
