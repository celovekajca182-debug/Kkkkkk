import json
import unittest
from unittest.mock import patch

import app


class BenchmarkGuardrailTests(unittest.TestCase):
    def setUp(self):
        app._auth_attempts.clear()

    def test_user_rate_limit_blocks_after_budget(self):
        with patch("app.time.monotonic", side_effect=[100.0, 101.0, 102.0]):
            app.enforce_user_rate_limit(7, "chat", 2)
            app.enforce_user_rate_limit(7, "chat", 2)
            with self.assertRaises(app.HTTPException) as caught:
                app.enforce_user_rate_limit(7, "chat", 2)
        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.headers["Retry-After"], "300")

    def test_user_rate_limits_are_isolated_by_account_and_action(self):
        with patch("app.time.monotonic", return_value=100.0):
            app.enforce_user_rate_limit(7, "chat", 1)
            app.enforce_user_rate_limit(8, "chat", 1)
            app.enforce_user_rate_limit(7, "code_assist", 1)
            with self.assertRaises(app.HTTPException):
                app.enforce_user_rate_limit(7, "chat", 1)

    def test_expired_rate_limit_entries_are_removed(self):
        with patch("app.time.monotonic", side_effect=[100.0, 401.0]):
            app.enforce_rate_limit_key("probe", 1, window_seconds=300)
            app.enforce_rate_limit_key("probe", 1, window_seconds=300)

    def test_health_reports_actual_model_provider_configuration(self):
        def configured(role=None):
            return role in (None, "chat", "code")

        with (
            patch.object(app, "provider_configured", side_effect=configured),
            patch.object(app, "transformer_model", None),
        ):
            payload = app.health()

        self.assertTrue(payload["llm_provider_enabled"])
        self.assertTrue(payload["external_model_api"])
        self.assertTrue(payload["chat_provider_enabled"])
        self.assertTrue(payload["code_provider_enabled"])
        self.assertEqual(payload["reasoning_model"], app.CHAT_MODEL)
        self.assertEqual(payload["coding_model"], app.CODE_MODEL)

    def test_public_capabilities_never_returns_provider_secrets(self):
        payload = json.dumps(app.capabilities(), ensure_ascii=False).lower()
        self.assertNotIn("api_key", payload)
        self.assertNotIn("hf_", payload)


if __name__ == "__main__":
    unittest.main()
