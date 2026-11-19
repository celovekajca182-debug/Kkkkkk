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

    def test_restore_device_backup_only_into_empty_server_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatHistoryStore(Path(directory) / "history.sqlite3")
            backup = [
                {"role": "user", "content": "Привет"},
                {"role": "assistant", "content": "Привет!"},
            ]
            self.assertTrue(store.restore_messages(3, "phone", backup))
            self.assertFalse(store.restore_messages(3, "phone", [
                {"role": "user", "content": "Подменённый вопрос"},
                {"role": "assistant", "content": "Подменённый ответ"},
            ]))
            self.assertEqual(store.turns(3, "phone"), [("Привет", "Привет!")])

    def test_clear_removes_only_selected_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ChatHistoryStore(Path(directory) / "history.sqlite3")
            store.save_turn(1, "one", "q1", "a1")
            store.save_turn(1, "two", "q2", "a2")
            removed = store.clear(1, "one")
            self.assertEqual(removed, 2)
            self.assertEqual(store.messages(1, "one"), [])
            self.assertEqual(len(store.messages(1, "two")), 2)

    def test_user_memories_persist_and_are_isolated_by_account(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite3"
            first = ChatHistoryStore(path)
            first.save_memory(1, "pref-1", "Предпочитаю русский язык", "preference")
            first.save_memory(1, "project-1", "Работаю над SWIN AI", "project")
            first.save_memory(2, "pref-1", "English only", "preference")

            reopened = ChatHistoryStore(path)
            first_items = reopened.memories(1)
            second_items = reopened.memories(2)
            self.assertEqual(len(first_items), 2)
            self.assertEqual(len(second_items), 1)
            self.assertEqual(second_items[0]["value"], "English only")

            reopened.save_memory(1, "pref-1", "Русский язык по умолчанию", "preference")
            self.assertEqual(len(reopened.memories(1)), 2)
            updated_preference = next(item for item in reopened.memories(1) if item["memory_key"] == "pref-1")
            self.assertEqual(updated_preference["value"], "Русский язык по умолчанию")

            self.assertEqual(reopened.delete_memory(1, "project-1"), 1)
            self.assertEqual(len(reopened.memories(1)), 1)
            self.assertEqual(reopened.delete_memory(1), 1)
            self.assertEqual(reopened.memories(1), [])
            self.assertEqual(len(reopened.memories(2)), 1)



if __name__ == "__main__":
    unittest.main()
