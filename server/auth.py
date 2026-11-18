"""Email/password accounts with Argon2id password hashes and opaque bearer sessions."""
from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

from pwdlib import PasswordHash

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PASSWORD_HASHER = PasswordHash.recommended()
TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60


def normalize_email(email: str) -> str:
    return email.strip().lower()


def validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if len(normalized) > 254 or not EMAIL_RE.fullmatch(normalized):
        raise ValueError("Укажи корректный адрес электронной почты.")
    return normalized


def validate_password(password: str) -> None:
    if len(password) < 12:
        raise ValueError("Пароль должен содержать минимум 12 символов.")
    if len(password) > 128 or len(password.encode("utf-8")) > 1024:
        raise ValueError("Пароль слишком длинный (максимум 128 символов).")


def hash_password(password: str) -> str:
    validate_password(password)
    return PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return PASSWORD_HASHER.verify(password, password_hash)
    except (ValueError, TypeError):
        return False


def digest_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    """Small SQLite account store; only password hashes and token hashes are persisted."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_id
                    ON auth_tokens(user_id);
                CREATE INDEX IF NOT EXISTS idx_auth_tokens_expiry
                    ON auth_tokens(expires_at);
                """
            )

    def create_user(self, email: str, password: str) -> dict[str, Any]:
        normalized = validate_email(email)
        encoded = hash_password(password)
        now = int(time.time())
        try:
            with self._connect() as db:
                cursor = db.execute(
                    "INSERT INTO users(email, password_hash, created_at) VALUES (?, ?, ?)",
                    (normalized, encoded, now),
                )
                return {"id": int(cursor.lastrowid), "email": normalized}
        except sqlite3.IntegrityError as error:
            if "users.email" in str(error).lower() or "unique" in str(error).lower():
                raise ValueError("Аккаунт с такой почтой уже существует.") from error
            raise

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        normalized = normalize_email(email)
        with self._connect() as db:
            row = db.execute(
                "SELECT id, email, password_hash FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return {"id": int(row["id"]), "email": row["email"]}

    def issue_token(self, user_id: int) -> tuple[str, int]:
        token = secrets.token_urlsafe(36)
        now = int(time.time())
        expires_at = now + TOKEN_TTL_SECONDS
        with self._connect() as db:
            db.execute(
                "INSERT INTO auth_tokens(token_hash, user_id, expires_at, created_at) "
                "VALUES (?, ?, ?, ?)",
                (digest_token(token), user_id, expires_at, now),
            )
            # Prune expired bearer tokens during new logins.
            db.execute("DELETE FROM auth_tokens WHERE expires_at <= ?", (now,))
        return token, expires_at

    def user_for_token(self, token: str) -> dict[str, Any] | None:
        if not token or len(token) > 256:
            return None
        now = int(time.time())
        with self._connect() as db:
            row = db.execute(
                """
                SELECT users.id, users.email
                FROM auth_tokens
                JOIN users ON users.id = auth_tokens.user_id
                WHERE auth_tokens.token_hash = ? AND auth_tokens.expires_at > ?
                """,
                (digest_token(token), now),
            ).fetchone()
        if row is None:
            return None
        return {"id": int(row["id"]), "email": row["email"]}

    def revoke_token(self, token: str) -> None:
        if not token:
            return
        with self._connect() as db:
            db.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (digest_token(token),))
