import unittest

from assistant_core import (
    search_web,
    search_web_many,
    build_reasoning_prompt,
    build_search_queries,
    is_safe_public_url,
    memory_category,
    memory_key_for,
    parse_memory_command,
    select_relevant_memories,
    should_research,
    validate_memory_value,
)


class WordTokenizer:
    def encode(self, value):
        return value.split()


class AssistantCoreTests(unittest.TestCase):
    def test_remote_search_is_disabled(self):
        self.assertEqual(search_web("current news"), [])
        self.assertEqual(search_web_many("Поищи последние новости"), [])

    def test_search_intent_is_detected_for_explicit_and_current_requests(self):
        self.assertTrue(should_research("Поищи последние новости по Godot 4 сегодня"))
        self.assertTrue(should_research("Find the current release date"))
        self.assertFalse(should_research("Привет, как дела?"))
        self.assertFalse(should_research("Объясни, что такое цикл for"))

    def test_search_query_expansion_removes_command_filler(self):
        queries = build_search_queries("Поищи последние новости о Godot 4")
        self.assertEqual(len(queries), 2)
        self.assertIn("Поищи", queries[0])
        self.assertEqual(queries[1], "последние новости о Godot 4")

    def test_memory_retrieval_prefers_relevant_project_facts(self):
        memories = [
            {"value": "Предпочитаю русский язык", "category": "preference"},
            {"value": "Мой проект использует Godot 4 и NavigationRegion3D", "category": "project"},
            {"value": "Люблю путешествовать", "category": "note"},
        ]
        selected = select_relevant_memories("Как настроить NavigationRegion3D в Godot?", memories)
        self.assertEqual(selected[0]["category"], "project")

    def test_search_url_filter_rejects_non_https_and_local_addresses(self):
        self.assertTrue(is_safe_public_url("https://docs.python.org/3/library/"))
        self.assertFalse(is_safe_public_url("http://docs.python.org/"))
        self.assertFalse(is_safe_public_url("https://127.0.0.1/admin"))
        self.assertFalse(is_safe_public_url("https://localhost/"))
        self.assertFalse(is_safe_public_url("https://user:pass@example.com/"))

    def test_explicit_memory_commands(self):
        self.assertEqual(
            parse_memory_command("Запомни: предпочитаю русский язык"),
            {"action": "save", "value": "предпочитаю русский язык"},
        )
        self.assertEqual(
            parse_memory_command("Что ты помнишь обо мне?"),
            {"action": "list"},
        )
        self.assertEqual(
            parse_memory_command("Забудь всё, что ты запомнила"),
            {"action": "forget_all"},
        )
        self.assertEqual(
            parse_memory_command("Забудь: предпочитаю русский язык"),
            {"action": "forget", "query": "предпочитаю русский язык"},
        )
        self.assertIsNone(parse_memory_command("Я предпочитаю русский язык"))

    def test_memory_secrets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Пароли"):
            validate_memory_value("Запомни мой пароль password для аккаунта")
        with self.assertRaisesRegex(ValueError, "500"):
            validate_memory_value("x" * 501)

    def test_memory_keys_are_stable_and_categories_are_inferred(self):
        self.assertEqual(memory_key_for("  Люблю   Python "), memory_key_for("люблю python"))
        self.assertEqual(memory_category("Предпочитаю русский язык"), "preference")
        self.assertEqual(memory_category("Работаю над проектом SWIN"), "project")

    def test_reasoning_prompt_uses_memory_and_retrieved_evidence(self):
        prompt = build_reasoning_prompt(
            history=[("Старый вопрос", "Старый ответ")],
            message="Какая версия библиотеки нужна?",
            language="ru",
            memories=[{"value": "Пользователь пишет приложение на Python"}],
            references=[{
                "title": "Официальная документация",
                "url": "https://docs.python.org/",
                "snippet": "Документация описывает текущий API.",
                "source_type": "Официальный источник",
            }],
            tokenizer=WordTokenizer(),
            block_size=512,
        )
        self.assertIn("Пользователь пишет приложение на Python", prompt)
        self.assertIn("Документация описывает текущий API", prompt)
        self.assertIn("Какая версия библиотеки нужна?", prompt)
        self.assertIn("do not print private reasoning", prompt)
        self.assertIn("Старый вопрос", prompt)


if __name__ == "__main__":
    unittest.main()
