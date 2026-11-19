import unittest

from language_support import (
    SUPPORTED_CHAT_LANGUAGES,
    detect_response_language,
    format_chat_prompt,
    localized_fallback,
    localized_small_talk,
    resolve_response_language,
)


class LanguageSupportTests(unittest.TestCase):
    def test_language_detection_for_all_supported_languages(self):
        samples = {
            "ru": "Привет, как дела? Давай напишем программу.",
            "uk": "Привіт! Як справи? Давай напишемо програму.",
            "en": "Hello, can you help me write a program?",
            "de": "Guten Tag, ich möchte ein Programm schreiben.",
            "es": "Hola, quiero escribir un programa. ¿Puedes ayudarme?",
            "fr": "Bonjour, je voudrais écrire un programme. Merci.",
            "zh": "你好，我想编写一个程序。你能帮助我吗？",
            "pt": "Olá, você pode me ajudar a escrever um programa?",
            "it": "Ciao, puoi aiutarmi a scrivere un programma?",
            "ja": "こんにちは。プログラムを書くのを手伝ってください。",
        }
        for expected, sample in samples.items():
            with self.subTest(language=expected):
                self.assertEqual(detect_response_language(sample), expected)

    def test_language_aliases_and_explicit_override(self):
        self.assertEqual(resolve_response_language("Hello", "Deutsch"), "de")
        self.assertEqual(resolve_response_language("Hello", "pt-BR"), "en")
        self.assertEqual(resolve_response_language("こんにちは", "ja"), "ja")

    def test_auto_uses_message_language(self):
        self.assertEqual(resolve_response_language("Olá, tudo bem?", "auto"), "pt")
        self.assertEqual(resolve_response_language("Как дела?", "auto"), "ru")

    def test_prompt_carries_response_language_instruction(self):
        prompt = format_chat_prompt([], "How are you?", "en")
        self.assertTrue(prompt.startswith("Response language: English."))
        self.assertIn("User: How are you?", prompt)
        self.assertTrue(prompt.endswith("Assistant:"))

    def test_small_talk_and_fallback_are_localized(self):
        self.assertIn("SWIN AI", localized_small_talk("Hello!", "en"))
        self.assertIn("SWIN AI", localized_small_talk("Olá!", "pt"))
        self.assertIn("SWIN AI", localized_small_talk("こんにちは", "ja"))
        self.assertIn("信頼できる", localized_fallback("ja"))
        self.assertIn("confiável", localized_fallback("pt"))

    def test_language_catalog_has_no_empty_labels(self):
        self.assertEqual(
            set(SUPPORTED_CHAT_LANGUAGES),
            {"auto", "ru", "en", "de", "es", "fr", "zh", "uk", "pt", "it", "ja"},
        )
        self.assertTrue(all(label for label in SUPPORTED_CHAT_LANGUAGES.values()))


if __name__ == "__main__":
    unittest.main()
