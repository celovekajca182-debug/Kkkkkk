"""Language selection and prompt formatting shared by SWIN training and inference.

The supported chat languages are deliberately limited to languages represented in
the curated instruction corpus. Auto-detection is heuristic; users can override it.
"""
from __future__ import annotations

import re
import unicodedata

SUPPORTED_CHAT_LANGUAGES = {
    "auto": "Автоопределение",
    "ru": "Русский",
    "en": "English",
    "de": "Deutsch",
    "es": "Español",
    "fr": "Français",
    "zh": "中文",
    "uk": "Українська",
}

LANGUAGE_NAMES_EN = {
    "ru": "Russian",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "zh": "Chinese",
    "uk": "Ukrainian",
}

LANGUAGE_ALIASES = {
    "russian": "ru", "русский": "ru", "русском": "ru", "ru-ru": "ru",
    "english": "en", "английский": "en", "английском": "en", "en-us": "en",
    "german": "de", "deutsch": "de", "немецкий": "de", "немецком": "de",
    "spanish": "es", "español": "es", "испанский": "es", "испанском": "es",
    "french": "fr", "français": "fr", "французский": "fr", "французском": "fr",
    "chinese": "zh", "中文": "zh", "китайский": "zh", "китайском": "zh",
    "ukrainian": "uk", "українська": "uk", "украинский": "uk", "украинском": "uk",
}

def normalize_response_language(value: str | None) -> str:
    candidate = (value or "auto").strip().lower()
    candidate = LANGUAGE_ALIASES.get(candidate, candidate)
    return candidate if candidate in SUPPORTED_CHAT_LANGUAGES else "auto"


def detect_response_language(text: str) -> str:
    """Best-effort detection for the supported languages, not a full language ID model."""
    value = (text or "").strip().lower()
    if not value:
        return "ru"

    # Script-specific languages first.
    if re.search(r"[іїєґ]", value):
        return "uk"
    if re.search(r"[а-яё]", value):
        return "ru"
    if re.search(r"[\u4e00-\u9fff]", value):
        return "zh"
    if re.search(r"[\u3040-\u30ff]", value):
        return "zh"  # Japanese is not yet part of the curated chat-SFT set.

    words = set(re.findall(r"[a-zà-ÿ]+", value))
    if re.search(r"[äöüß]", value) or words.intersection({"der", "die", "das", "und", "nicht", "ich", "bitte", "danke"}):
        return "de"
    if re.search(r"[¿¡ñ]", value) or words.intersection({"hola", "gracias", "por", "para", "cómo", "que", "buenos", "buenas"}):
        return "es"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", value) or words.intersection({"bonjour", "merci", "avec", "pour", "comment", "est", "une", "les"}):
        return "fr"
    return "en"


def resolve_response_language(message: str, requested: str | None = "auto") -> str:
    normalized = normalize_response_language(requested)
    return detect_response_language(message) if normalized == "auto" else normalized


def response_language_instruction(language: str) -> str:
    normalized = normalize_response_language(language)
    if normalized == "auto":
        normalized = "ru"
    return f"Response language: {LANGUAGE_NAMES_EN[normalized]}."


def format_chat_prompt(
    history: list[tuple[str, str]],
    message: str,
    language: str,
) -> str:
    """The exact chat header used for both multilingual SFT and inference."""
    parts = [response_language_instruction(language)]
    for old_user, old_answer in history[-6:]:
        parts.append(f"User: {old_user.strip()}\nAssistant: {old_answer.strip()}")
    parts.append(f"User: {message.strip()}\nAssistant:")
    return "\n".join(parts)


def format_training_prompt(question: str, language: str) -> str:
    return f"{response_language_instruction(language)}\nUser: {question.strip()}\nAssistant:"


def localized_small_talk(message: str, language: str) -> str | None:
    value = (message or "").strip().casefold()
    language = normalize_response_language(language)
    if language == "auto":
        language = detect_response_language(value)

    greetings = {
        "ru": ("привет", "здравств", "доброе утро", "добрый день", "добрый вечер"),
        "en": ("hello", "hi", "hey", "good morning", "good afternoon", "good evening"),
        "de": ("hallo", "guten morgen", "guten tag", "guten abend", "hi"),
        "es": ("hola", "buenos días", "buenas tardes", "buenas noches"),
        "fr": ("bonjour", "bonsoir", "salut", "coucou"),
        "zh": ("你好", "您好", "早上好", "晚上好"),
        "uk": ("привіт", "добрий день", "доброго ранку", "добрий вечір"),
    }
    greeting_replies = {
        "ru": "Привет! Я SWIN AI. Чем могу помочь?",
        "en": "Hello! I'm SWIN AI. How can I help?",
        "de": "Hallo! Ich bin SWIN AI. Wie kann ich helfen?",
        "es": "¡Hola! Soy SWIN AI. ¿Cómo puedo ayudarte?",
        "fr": "Bonjour ! Je suis SWIN AI. Comment puis-je vous aider ?",
        "zh": "你好！我是 SWIN AI。有什么可以帮你？",
        "uk": "Привіт! Я SWIN AI. Чим можу допомогти?",
    }
    normalized = re.sub(r"[!?.,。！？，]+$", "", value).strip()
    if any(normalized.startswith(greeting) for greeting in greetings[language]):
        return greeting_replies[language]

    how_are_you = {
        "ru": ("как дела", "как ты", "как поживаешь"),
        "en": ("how are you", "how's it going", "how do you do"),
        "de": ("wie geht", "wie geht es dir"),
        "es": ("cómo estás", "que tal"),
        "fr": ("comment allez-vous", "ça va", "comment vas-tu"),
        "zh": ("你好吗", "怎么样", "最近好吗"),
        "uk": ("як справи", "як ти", "як поживаєш"),
    }
    replies = {
        "ru": "Работаю и готов помочь с задачей. А что будем делать?",
        "en": "I'm ready to help with your task. What shall we work on?",
        "de": "Ich bin bereit, bei deiner Aufgabe zu helfen. Woran arbeiten wir?",
        "es": "Estoy lista para ayudarte. ¿En qué trabajamos?",
        "fr": "Je suis prêt à vous aider. Sur quoi allons-nous travailler ?",
        "zh": "我准备好帮你了。我们接下来做什么？",
        "uk": "Готовий допомогти з твоїм завданням. Що будемо робити?",
    }
    if any(phrase in normalized for phrase in how_are_you[language]):
        return replies[language]
    return None


def localized_fallback(language: str) -> str:
    normalized = normalize_response_language(language)
    if normalized == "auto":
        normalized = "ru"
    messages = {
        "ru": "Пока не могу надёжно сформировать ответ на этом языке. Попробуй уточнить вопрос или выбрать другой язык ответа.",
        "en": "I can't produce a reliable answer in this language yet. Please clarify the question or choose another response language.",
        "de": "Ich kann derzeit noch keine zuverlässige Antwort in dieser Sprache geben. Bitte präzisiere die Frage oder wähle eine andere Antwortsprache.",
        "es": "Todavía no puedo generar una respuesta fiable en este idioma. Aclara la pregunta o elige otro idioma de respuesta.",
        "fr": "Je ne peux pas encore produire une réponse fiable dans cette langue. Précisez la question ou choisissez une autre langue.",
        "zh": "我目前还不能可靠地用这种语言回答。请补充问题细节或选择其他回复语言。",
        "uk": "Поки не можу надійно сформувати відповідь цією мовою. Уточни запитання або вибери іншу мову відповіді.",
    }
    return messages[normalized]
