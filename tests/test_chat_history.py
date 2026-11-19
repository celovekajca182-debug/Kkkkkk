import tempfile
import unittest
from pathlib import Path

from chat_history import ChatHistoryStore


class ChatHistoryStoreTests(unittest.TestCase):
    def test_turns_survive_store_recreation_and_are_user_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "swin.sqlite3"
            first = ChatHistoryStore(path)
            first.save_turn(1, "phone", "Привет", "Здравствуйте!")
            first.save_turn(1, "phone", "Как дела?", "Готов помогать.")
            first.save_turn(2, "phone", "Секретный вопрос", "Ответ другого пользователя.")

            reopened = ChatHistoryStore(path)
            self.assertEqual(
                reopened.turns(1, "phone"),
                [("Привет", "Здравствуйте!"), ("Как дела?", "Готов помогать.")],
            )
            self.assertEqual(len(reopened.messages(1, "phone")), 4)
            self.assertEqual(len(reopened.messages(2, "phone")), 2)

    def test_history_is_bounded_and_returns_chronological_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatHistoryStore(Path(directory) / "history.sqlite3", max_messages_per_session=20)
            for index in range(15):
                store.save_turn(7, "session", f"Вопрос {index}", f"Ответ {index}")

            messages = store.messages(7, "session", limit=80)
            self.assertEqual(len(messages), 20)
            self.assertEqual(messages[0]["content"], "Вопрос 5")
            self.assertEqual(messages[-1]["content"], "Ответ 14")

    def test_clear_removes_only_selected_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatHistoryStore(Path(directory) / "history.sqlite3")
            store.save_turn(1, "one", "q1", "a1")
            store.save_turn(1, "two", "q2", "a2")
            removed = store.clear(1, "one")
            self.assertEqual(removed, 2)
            self.assertEqual(store.messages(1, "one"), [])
            self.assertEqual(len(store.messages(1, "two")), 2)


if __name__ == "__main__":
    unittest.main()
