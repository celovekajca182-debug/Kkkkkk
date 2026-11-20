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

    def test_hugging_face_router_requires_a_server_token(self):
        with (
            patch.object(llm_provider, "BASE_URL", "https://router.huggingface.co/v1"),
            patch.object(llm_provider, "API_KEY", ""),
            patch.object(llm_provider, "CHAT_BASE_URL", ""),
            patch.object(llm_provider, "CODE_BASE_URL", ""),
        ):
            self.assertFalse(llm_provider.provider_configured("chat"))
            self.assertFalse(llm_provider.provider_configured("code"))
            self.assertFalse(llm_provider.provider_configured())
            with self.assertRaisesRegex(RuntimeError, "token is missing"):
                llm_provider.generate_model_text("hello", role="chat")
            with self.assertRaisesRegex(RuntimeError, "endpoint is not configured"):
                llm_provider.generate_model_text("write code", role="code")

    def test_shared_hugging_face_token_enables_both_model_roles(self):
        with (
            patch.object(llm_provider, "BASE_URL", "https://router.huggingface.co/v1"),
            patch.object(llm_provider, "API_KEY", "hf-test-token"),
            patch.object(llm_provider, "CHAT_BASE_URL", ""),
            patch.object(llm_provider, "CODE_BASE_URL", ""),
        ):
            self.assertTrue(llm_provider.provider_configured("chat"))
            self.assertFalse(llm_provider.provider_configured("code"))

    def test_self_hosted_endpoint_can_be_configured_without_a_key(self):
        with (
            patch.object(llm_provider, "BASE_URL", "https://inference.example/v1"),
            patch.object(llm_provider, "API_KEY", ""),
            patch.object(llm_provider, "CHAT_BASE_URL", ""),
            patch.object(llm_provider, "CODE_BASE_URL", ""),
            patch.object(llm_provider, "CODE_MODEL", "example/coder"),
        ):
            self.assertTrue(llm_provider.provider_configured("chat"))
            self.assertTrue(llm_provider.provider_configured("code"))

    def test_custom_adapter_requires_dedicated_code_endpoint(self):
        with (
            patch.object(llm_provider, "BASE_URL", "https://router.huggingface.co/v1"),
            patch.object(llm_provider, "API_KEY", "hf-test-token"),
            patch.object(llm_provider, "CODE_BASE_URL", ""),
            patch.object(llm_provider, "CODE_MODEL", llm_provider.SWIN_ADAPTER_MODEL),
        ):
            self.assertFalse(llm_provider.provider_configured("code"))
        with (
            patch.object(llm_provider, "CODE_BASE_URL", "https://gpu.example/v1"),
            patch.object(llm_provider, "CODE_MODEL", llm_provider.SWIN_ADAPTER_MODEL),
        ):
            self.assertTrue(llm_provider.provider_configured("code"))

    def test_model_roles_are_separate(self):
        self.assertEqual(
            llm_provider.model_for_role("chat"),
            "Qwen/Qwen3-4B-Thinking-2507",
        )
        self.assertEqual(
            llm_provider.model_for_role("code"),
            "MAKSIM67771488/SWIN-AI-Coder-Smart",
        )

    def test_completion_uses_correct_model_and_auth(self):
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "print(2 + 2)"}}]
        }
        with (
            patch.object(llm_provider, "BASE_URL", "https://inference.example/v1"),
            patch.object(llm_provider, "API_KEY", "test-secret"),
            patch.object(llm_provider, "CODE_MODEL", "example/coder"),
            patch.object(llm_provider, "CODE_BASE_URL", ""),
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

    def test_role_specific_endpoints_override_shared_gateway(self):
        payload = {"choices": [{"message": {"content": "ok"}}]}
        with (
            patch.object(llm_provider, "BASE_URL", "https://shared.example/v1"),
            patch.object(llm_provider, "CHAT_BASE_URL", "https://chat.example/v1"),
            patch.object(llm_provider, "CODE_BASE_URL", "https://code.example/v1"),
            patch.object(llm_provider, "CHAT_API_KEY", "chat-secret"),
            patch.object(llm_provider, "CODE_API_KEY", "code-secret"),
            patch(
                "llm_provider.urllib.request.urlopen",
                side_effect=[FakeResponse(payload), FakeResponse(payload)],
            ) as urlopen,
        ):
            self.assertTrue(llm_provider.provider_configured("chat"))
            self.assertTrue(llm_provider.provider_configured("code"))
            self.assertTrue(llm_provider.provider_configured())
            llm_provider.generate_model_text("chat prompt", role="chat")
            llm_provider.generate_model_text("code prompt", role="code")

        chat_request, code_request = [call.args[0] for call in urlopen.call_args_list]
        self.assertEqual(chat_request.full_url, "https://chat.example/v1/chat/completions")
        self.assertEqual(code_request.full_url, "https://code.example/v1/chat/completions")
        self.assertEqual(chat_request.get_header("Authorization"), "Bearer chat-secret")
        self.assertEqual(code_request.get_header("Authorization"), "Bearer code-secret")

    def test_thinking_trace_is_removed_from_visible_answer(self):
        self.assertEqual(
            llm_provider._remove_private_reasoning("private trace</think>Final answer"),
            "Final answer",
        )
        self.assertEqual(
            llm_provider._remove_private_reasoning("<think>private</think>Answer"),
            "Answer",
        )
        self.assertEqual(
            llm_provider._remove_private_reasoning("Plain answer"),
            "Plain answer",
        )
        self.assertEqual(
            llm_provider._remove_private_reasoning("Answer <think>unfinished private trace"),
            "Answer",
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
