import unittest

from app import looks_like_garbage, safe_small_talk


class ChatSafetyTests(unittest.TestCase):
    def test_small_talk_returns_russian_answer(self):
        answer = safe_small_talk("как дела?")
        self.assertIsNotNone(answer)
        self.assertIn("готов помочь", answer)

    def test_greeting_is_handled_without_model_generation(self):
        answer = safe_small_talk("Привет!")
        self.assertTrue(answer.startswith("Привет!"))

    def test_unknown_prompt_is_not_misclassified_as_small_talk(self):
        self.assertIsNone(safe_small_talk("Как исправить ошибку в Python?"))

    def test_repeated_byte_level_gibberish_is_rejected(self):
        sample = "ооооеоиеооооооооооииоооееоооееиоо"
        self.assertTrue(looks_like_garbage(sample))

    def test_unicode_replacement_character_is_rejected(self):
        self.assertTrue(looks_like_garbage("непонятный текст � после генерации"))

    def test_long_unbroken_model_noise_is_rejected(self):
        sample = "оооооеиеооооооооооииоооеооееиеио"
        self.assertTrue(looks_like_garbage(sample))

    def test_repeated_vowel_stream_is_rejected(self):
        self.assertTrue(looks_like_garbage("оооооооооооооооооооооооо"))

    def test_normal_russian_sentence_is_not_rejected(self):
        sample = "Для этого открой настройки приложения и проверь подключение к серверу."
        self.assertFalse(looks_like_garbage(sample))


if __name__ == "__main__":
    unittest.main()
