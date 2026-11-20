import unittest
from unittest.mock import patch

from fastapi import HTTPException

import peft_inference_server as inference


class FakeTokenizer:
    chat_template = "fake-template"

    def __init__(self):
        self.received = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.received = (messages, tokenize, add_generation_prompt)
        return "rendered prompt"


class PeftInferenceContractTests(unittest.TestCase):
    def test_endpoint_serves_only_the_swin_adapter_id(self):
        self.assertEqual(inference._check_model(None), inference.MODEL_ID)
        self.assertEqual(inference._check_model(inference.MODEL_ID), inference.MODEL_ID)
        self.assertEqual(inference._check_model("swin-ai-coder"), inference.MODEL_ID)
        with self.assertRaises(HTTPException) as error:
            inference._check_model("Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(error.exception.status_code, 400)

    def test_uses_adapter_chat_template_and_message_roles(self):
        tokenizer = FakeTokenizer()
        messages = [
            inference.ChatMessage(role="system", content="You are SWIN."),
            inference.ChatMessage(role="user", content="Write a function."),
        ]
        prompt = inference._apply_chat_template(tokenizer, messages)
        self.assertEqual(prompt, "rendered prompt")
        self.assertEqual(tokenizer.received[0][0]["role"], "system")
        self.assertEqual(tokenizer.received[0][1]["role"], "user")
        self.assertFalse(tokenizer.received[1])
        self.assertTrue(tokenizer.received[2])

    def test_missing_or_invalid_server_key_is_rejected(self):
        with patch.object(inference, "INFERENCE_API_KEY", "secret"):
            with self.assertRaises(HTTPException) as missing:
                inference._authorize(None)
            self.assertEqual(missing.exception.status_code, 401)
            with self.assertRaises(HTTPException) as wrong:
                inference._authorize("Bearer incorrect")
            self.assertEqual(wrong.exception.status_code, 401)
            inference._authorize("Bearer secret")

    def test_missing_server_key_fails_closed(self):
        with patch.object(inference, "INFERENCE_API_KEY", ""):
            with self.assertRaises(HTTPException) as error:
                inference._authorize("Bearer any-value")
        self.assertEqual(error.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
