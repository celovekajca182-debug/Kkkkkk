import tempfile
import unittest
from pathlib import Path

from postgres_stores import _connect, migrate_sqlite_to_postgres


class PostgresStoreTests(unittest.TestCase):
    def test_connection_rejects_non_postgres_urls(self):
        with self.assertRaisesRegex(ValueError, "PostgreSQL"):
            _connect("sqlite:///tmp/swin.sqlite3")

    def test_missing_sqlite_source_is_a_safe_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            result = migrate_sqlite_to_postgres(
                Path(directory) / "not-created.sqlite3",
                "postgresql://invalid.example/database",
            )
        self.assertEqual(result, {"users": 0, "tokens": 0, "messages": 0})

    def test_empty_sqlite_source_is_a_safe_migration_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.sqlite3"
            source.touch()
            result = migrate_sqlite_to_postgres(
                source,
                "postgresql://invalid.example/database",
            )
        self.assertEqual(result, {"users": 0, "tokens": 0, "messages": 0})


if __name__ == "__main__":
    unittest.main()
