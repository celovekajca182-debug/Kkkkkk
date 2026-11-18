import tempfile
import unittest
from pathlib import Path

from auth import AuthStore, digest_token, validate_email, validate_password


class AuthStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "auth.sqlite3"
        self.store = AuthStore(self.database)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_email_is_normalized_and_password_is_not_stored_plaintext(self):
        user = self.store.create_user("  USER@Example.COM ", "correct horse battery")
        self.assertEqual(user["email"], "user@example.com")
        import sqlite3

        with sqlite3.connect(self.database) as db:
            stored_hash = db.execute(
                "SELECT password_hash FROM users WHERE id = ?", (user["id"],)
            ).fetchone()[0]
        self.assertNotEqual(stored_hash, "correct horse battery")
        self.assertTrue(stored_hash.startswith("$argon2id$"))
        self.assertEqual(self.store.authenticate("USER@example.com", "correct horse battery"), user)

    def test_duplicate_email_is_rejected(self):
        self.store.create_user("user@example.com", "correct horse battery")
        with self.assertRaisesRegex(ValueError, "уже существует"):
            self.store.create_user("USER@example.com", "different password")

    def test_bearer_token_is_opaque_stored_as_digest_and_revocable(self):
        user = self.store.create_user("user@example.com", "correct horse battery")
        token, expires_at = self.store.issue_token(user["id"])
        self.assertNotEqual(token, digest_token(token))
        self.assertEqual(self.store.user_for_token(token), user)
        self.assertGreater(expires_at, 0)
        self.store.revoke_token(token)
        self.assertIsNone(self.store.user_for_token(token))

    def test_password_and_email_validation(self):
        self.assertEqual(validate_email(" USER@example.com "), "user@example.com")
        with self.assertRaises(ValueError):
            validate_email("not-an-email")
        with self.assertRaises(ValueError):
            validate_password("short")


if __name__ == "__main__":
    unittest.main()
