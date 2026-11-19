"""SQLite-backed per-user chat history for SWIN AI."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


class ChatHistoryStore:
    """Persist chat turns by account and client session.

    The store uses the same SQLite database path as authentication, but only
    stores conversation text and timestamps; credentials are not duplicated.
    """

    def __init__(self, database_path: str | Path, max_messages_per_session: int = 400):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_messages_per_session = max(20, int(max_messages_per_session))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(user_id, session_id, id)
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    user_id INTEGER NOT NULL,
                    memory_key TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'note',
                    value TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(user_id, memory_key)
                )
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_memories_updated
                ON user_memories(user_id, updated_at DESC)
                """
            )

    def save_turn(
        self, user_id: int, session_id: str, user_message: str, assistant_message: str
    ) -> None:
        now = int(time.time())
        with self._connect() as db:
            db.executemany(
                """
                INSERT INTO chat_messages(user_id, session_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (int(user_id), session_id, "user", user_message, now),
                    (int(user_id), session_id, "assistant", assistant_message, now),
                ),
            )
            db.execute(
                """
                DELETE FROM chat_messages
                WHERE user_id = ? AND session_id = ?
                  AND id NOT IN (
                    SELECT id FROM chat_messages
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY id DESC LIMIT ?
                  )
                """,
                (
                    int(user_id),
                    session_id,
                    int(user_id),
                    session_id,
                    self.max_messages_per_session,
                ),
            )

    def messages(
        self, user_id: int, session_id: str, limit: int = 80
    ) -> list[dict[str, Any]]:
        safe_limit = max(2, min(int(limit), self.max_messages_per_session))
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (int(user_id), session_id, safe_limit),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": int(row["created_at"]),
            }
            for row in reversed(rows)
        ]

    def sessions(self, user_id: int, limit: int = 50) -> list[dict[str, Any]]:
        """List recent conversations for one account, with safe display previews."""
        safe_limit = max(1, min(int(limit), 100))
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT
                    session_id,
                    (
                        SELECT content FROM chat_messages AS first_message
                        WHERE first_message.user_id = messages.user_id
                          AND first_message.session_id = messages.session_id
                          AND first_message.role = 'user'
                        ORDER BY first_message.id ASC LIMIT 1
                    ) AS title,
                    (
                        SELECT content FROM chat_messages AS last_message
                        WHERE last_message.user_id = messages.user_id
                          AND last_message.session_id = messages.session_id
                        ORDER BY last_message.id DESC LIMIT 1
                    ) AS preview,
                    MAX(created_at) AS updated_at,
                    COUNT(*) AS message_count,
                    MAX(id) AS last_message_id
                FROM chat_messages AS messages
                WHERE user_id = ?
                GROUP BY user_id, session_id
                ORDER BY last_message_id DESC
                LIMIT ?
                """,
                (int(user_id), safe_limit),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "title": (row["title"] or "Новый чат")[:120],
                "preview": (row["preview"] or "")[:220],
                "updated_at": int(row["updated_at"]),
                "message_count": int(row["message_count"]),
            }
            for row in rows
        ]

    def turns(
        self, user_id: int, session_id: str, limit: int = 6
    ) -> list[tuple[str, str]]:
        rows = self.messages(user_id, session_id, limit=max(2, min(limit * 2, 24)))
        turns: list[tuple[str, str]] = []
        pending_user: str | None = None
        for row in rows:
            if row["role"] == "user":
                pending_user = row["content"]
            elif row["role"] == "assistant" and pending_user is not None:
                turns.append((pending_user, row["content"]))
                pending_user = None
        return turns[-max(1, int(limit)):]

    def restore_messages(
        self, user_id: int, session_id: str, messages: list[dict[str, str]]
    ) -> bool:
        """Restore a local device backup only when the server session is empty."""
        normalized = [
            (item.get("role", ""), item.get("content", "").strip())
            for item in messages
            if item.get("role") in {"user", "assistant"}
            and item.get("content", "").strip()
        ][-self.max_messages_per_session:]
        if not normalized:
            return False
        now = int(time.time())
        with self._connect() as db:
            existing = db.execute(
                "SELECT 1 FROM chat_messages WHERE user_id = ? AND session_id = ? LIMIT 1",
                (int(user_id), session_id),
            ).fetchone()
            if existing is not None:
                return False
            db.executemany(
                """
                INSERT INTO chat_messages(user_id, session_id, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (int(user_id), session_id, role, content[:120000], now)
                    for role, content in normalized
                ],
            )
        return True

    def save_memory(
        self, user_id: int, memory_key: str, value: str, category: str = "note"
    ) -> dict[str, Any]:
        """Insert or update an explicit, account-scoped memory fact."""
        now = int(time.time())
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO user_memories(user_id, memory_key, category, value, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, memory_key) DO UPDATE SET
                    category = excluded.category,
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (int(user_id), memory_key, category, value[:500], now, now),
            )
        return {
            "memory_key": memory_key,
            "category": category,
            "value": value[:500],
            "updated_at": now,
        }

    def memories(self, user_id: int, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 50))
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT memory_key, category, value, updated_at
                FROM user_memories
                WHERE user_id = ?
                ORDER BY updated_at DESC, memory_key
                LIMIT ?
                """,
                (int(user_id), safe_limit),
            ).fetchall()
        return [
            {
                "memory_key": row["memory_key"],
                "category": row["category"],
                "value": row["value"],
                "updated_at": int(row["updated_at"]),
            }
            for row in rows
        ]

    def delete_memory(self, user_id: int, memory_key: str | None = None) -> int:
        with self._connect() as db:
            if memory_key is None:
                cursor = db.execute(
                    "DELETE FROM user_memories WHERE user_id = ?",
                    (int(user_id),),
                )
            else:
                cursor = db.execute(
                    "DELETE FROM user_memories WHERE user_id = ? AND memory_key = ?",
                    (int(user_id), memory_key),
                )
            return int(cursor.rowcount)

    def clear(self, user_id: int, session_id: str) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "DELETE FROM chat_messages WHERE user_id = ? AND session_id = ?",
                (int(user_id), session_id),
            )
            return int(cursor.rowcount)
