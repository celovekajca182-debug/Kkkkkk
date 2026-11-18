import unittest

from app import conversation_memory, remember_turn


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self):
        conversation_memory.clear()

    def test_session_history_is_isolated(self):
        remember_turn("phone-a", "Привет", "Привет!")
        remember_turn("phone-b", "Как дела?", "Хорошо.")
        self.assertEqual(len(conversation_memory["phone-a"]), 1)
        self.assertEqual(len(conversation_memory["phone-b"]), 1)

    def test_history_keeps_only_last_six_turns(self):
        for index in range(10):
            remember_turn("phone-a", f"Вопрос {index}", f"Ответ {index}")
        turns = conversation_memory["phone-a"]
        self.assertEqual(len(turns), 6)
        self.assertEqual(turns[0][0], "Вопрос 4")
        self.assertEqual(turns[-1][0], "Вопрос 9")

    def test_session_count_is_bounded(self):
        for index in range(260):
            remember_turn(f"session-{index}", "q", "a")
        self.assertLessEqual(len(conversation_memory), 256)
        self.assertNotIn("session-0", conversation_memory)


if __name__ == "__main__":
    unittest.main()
