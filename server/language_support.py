"""Language selection and prompt formatting shared by SWIN training and inference.

Auto-detection is intentionally conservative and supports the languages present
in the curated multilingual instruction corpus. Users can always override it.
"""
from __future__ import annotations

import re

SUPPORTED_CHAT_LANGUAGES = {
    "auto": "Автоопределение",
    "ru": "Русский",
    "en": "English",
    "de": "Deutsch",
    "es": "Español",
    "fr": "Français",
    "zh": "中文",
    "uk": "Українська",
    "pt": "Português",
    "it": "Italiano",
    "ja": "日本語",
}

LANGUAGE_NAMES_EN = {
    "ru": "Russian",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "zh": "Chinese",
    "uk": "Ukrainian",
    "pt": "Portuguese",
    "it": "Italian",
    "ja": "Japanese",
}

LANGUAGE_ALIASES = {
    "russian": "ru", "русский": "ru", "русском": "ru", "ru-ru": "ru",
    "english": "en", "английский": "en", "английском": "en", "en-us": "en",
    "german": "de", "deutsch": "de", "немецкий": "de", "немецком": "de",
    "spanish": "es", "español": "es", "испанский": "es", "испанском": "es",
    "french": "fr", "français": "fr", "французский": "fr", "французском": "fr",
    "chinese": "zh", "中文": "zh", "китайский": "zh", "китайском": "zh",
    "ukrainian": "uk", "українська": "uk", "украинский": "uk", "украинском": "uk",
    "portuguese": "pt", "português": "pt", "португальский": "pt", "португальском": "pt",
    "italian": "it", "italiano": "it", "итальянский": "it", "итальянском": "it",
    "japanese": "ja", "日本語": "ja", "японский": "ja", "японском": "ja",
}


def normalize_response_language(value: str | None) -> str:
    candidate = (value or "auto").strip().lower().replace("_", "-")
    candidate = LANGUAGE_ALIASES.get(candidate, candidate)
    # Accept BCP-47 variants such as pt-BR, en-US and zh-Hans-CN.
    if candidate not in SUPPORTED_CHAT_LANGUAGES:
        candidate = candidate.split("-", 1)[0]
    return candidate if candidate in SUPPORTED_CHAT_LANGUAGES else "auto"


def detect_response_language(text: str) -> str:
    """Best-effort language ID for the ten supported chat languages."""
    value = (text or "").strip().lower()
    if not value:
        return "ru"

    # Distinguish scripts before applying Latin-language heuristics.
    if re.search(r"[іїєґ]", value):
        return "uk"
    if re.search(r"[а-яё]", value):
        return "ru"
    if (
        re.search(r"[\u3040-\u30ff]", value)
        or any(marker in value for marker in ("日本語", "ありがとう", "です", "ます", "ください"))
    ):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", value):
        return "zh"

    words = set(re.findall(r"[a-zà-ÿ]+", value))
    if (
        re.search(r"[äöüß]", value)
        or words.intersection({"der", "die", "das", "und", "nicht", "ich", "bitte", "danke", "ist"})
    ):
        return "de"
    if (
        re.search(r"[ãõ]", value)
        or words.intersection({"olá", "obrigado", "obrigada", "você", "vocês", "não", "bom", "obrigado"})
    ):
        return "pt"
    if (
        re.search(r"[¿¡ñ]", value)
        or words.intersection({"hola", "gracias", "buenos", "buenas", "también", "pero", "estoy"})
    ):
        return "es"
    if (
        re.search(r"[âêîôûœ]", value)
        or words.intersection({"bonjour", "merci", "avec", "pour", "comment", "une", "les", "vous"})
    ):
        return "fr"
    if (
        "perché" in value
        or "è" in value
        or words.intersection({"ciao", "grazie", "buongiorno", "buonasera", "sono", "come", "perche"})
    ):
        return "it"
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

    normalized = re.sub(r"[!?.,。！？，،]+$", "", value).strip()
    greetings = {
        "ru": ("привет", "здравств", "доброе утро", "добрый день", "добрый вечер"),
        "en": ("hello", "hi", "hey", "good morning", "good afternoon", "good evening"),
        "de": ("hallo", "guten morgen", "guten tag", "guten abend", "hi"),
        "es": ("hola", "buenos días", "buenas tardes", "buenas noches"),
        "fr": ("bonjour", "bonsoir", "salut", "coucou"),
        "zh": ("你好", "您好", "早上好", "晚上好"),
        "uk": ("привіт", "добрий день", "доброго ранку", "добрий вечір"),
        "pt": ("olá", "oi", "bom dia", "boa tarde", "boa noite"),
        "it": ("ciao", "buongiorno", "buonasera", "salve"),
        "ja": ("こんにちは", "こんばんは", "おはよう", "やあ"),
    }
    greeting_replies = {
        "ru": "Привет! Я SWIN AI. Чем могу помочь?",
        "en": "Hello! I'm SWIN AI. How can I help?",
        "de": "Hallo! Ich bin SWIN AI. Wie kann ich helfen?",
        "es": "¡Hola! Soy SWIN AI. ¿Cómo puedo ayudarte?",
        "fr": "Bonjour ! Je suis SWIN AI. Comment puis-je vous aider ?",
        "zh": "你好！我是 SWIN AI。有什么可以帮你？",
        "uk": "Привіт! Я SWIN AI. Чим можу допомогти?",
        "pt": "Olá! Sou a SWIN AI. Como posso ajudar?",
        "it": "Ciao! Sono SWIN AI. Come posso aiutarti?",
        "ja": "こんにちは！SWIN AIです。どのようにお手伝いできますか？",
    }
    if any(normalized == greeting or normalized.startswith(greeting + " ") for greeting in greetings[language]):
        return greeting_replies[language]

    how_are_you = {
        "ru": ("как дела", "как ты", "как поживаешь"),
        "en": ("how are you", "how's it going", "how do you do"),
        "de": ("wie geht", "wie geht es dir"),
        "es": ("cómo estás", "que tal", "qué tal"),
        "fr": ("comment allez-vous", "ça va", "comment vas-tu"),
        "zh": ("你好吗", "怎么样", "最近好吗"),
        "uk": ("як справи", "як ти", "як поживаєш"),
        "pt": ("como você está", "tudo bem", "como vai"),
        "it": ("come stai", "come va", "tutto bene"),
        "ja": ("元気ですか", "元気？", "調子はどう"),
    }
    replies = {
        "ru": "Работаю и готов помочь с задачей. А что будем делать?",
        "en": "I'm ready to help with your task. What shall we work on?",
        "de": "Ich bin bereit, bei deiner Aufgabe zu helfen. Woran arbeiten wir?",
        "es": "Estoy listo para ayudarte. ¿En qué trabajamos?",
        "fr": "Je suis prêt à vous aider. Sur quoi allons-nous travailler ?",
        "zh": "我准备好帮你了。我们接下来做什么？",
        "uk": "Готовий допомогти з твоїм завданням. Що будемо робити?",
        "pt": "Estou pronto para ajudar. No que vamos trabalhar?",
        "it": "Sono pronto ad aiutarti. Su cosa lavoriamo?",
        "ja": "お手伝いする準備ができています。何から始めましょうか？",
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
        "pt": "Ainda não consigo formular uma resposta confiável neste idioma. Esclareça a pergunta ou escolha outro idioma.",
        "it": "Non riesco ancora a formulare una risposta affidabile in questa lingua. Chiarisci la domanda o scegli un'altra lingua.",
        "ja": "この言語で信頼できる回答をまだ作成できません。質問を詳しくするか、別の言語を選んでください。",
    }
    return messages[normalized]


CODE_ASSIST_COPY = {
    "ru": {
        "plan": ["Подготовить решение на {language}.", "Сверить материалы и приоритетно использовать официальную документацию.", "Создать черновик и статически проверить его без запуска."],
        "summary": {"retrieved_example":"Найден близкий проверенный пример. Адаптируй его к проекту перед использованием.","model_not_loaded":"Поиск завершён, но веса SWIN не загружены. Код не генерировался.","generation_failed":"Не удалось запустить генерацию кода.","generation_rejected":"Бессвязный текст вместо кода отклонён. Уточни задачу или используй проверенный пример.","draft_validated":"Черновик создан и прошёл доступную статическую проверку.","draft_needs_review":"Черновик создан, но проверка ограничена или требует исправления."},
        "validation":{"passed":"Статическая проверка пройдена.","limited":"Доступна только ограниченная статическая проверка.","failed":"Статическая проверка не пройдена.","not_run":"Статическая проверка не выполнялась.","not_supported":"Для этого формата нет полноценной статической проверки."},
        "warnings":{"retrieved":"Это близкий пример, а не полностью адаптированная реализация.","adapt":"Сверь пути узлов, версии библиотек и входные данные со своим проектом.","limited":"Полноценный компилятор или парсер для этого языка недоступен.","failed":"Код не прошёл статическую проверку. Не используй его как готовое решение.","web":"Веб-источники могут содержать ошибки или устаревшие примеры.","static":"Статическая проверка не доказывает правильность алгоритма.","runtime":"Код и тесты не запускались на сервере.","deploy":"Для генерации разверни сервер с checkpoint SWIN.","resources":"Проверь загрузку модели и ресурсы сервера."},
        "research_found":"Найдены веб-источники; это не гарантия полноты или достоверности каждой страницы.","research_none":"Веб-источники не получены.","runtime":"Код и тесты не запускались: сервер не исполняет недоверенные программы.","cross_found":"К результату приложены материалы для ручной сверки.","cross_none":"Материалы для сверки не получены.","empty":"Опиши задачу, для которой нужен код.","too_long":"Описание задачи слишком длинное (максимум 4000 символов).",
        "activity":{"requirements":"Требования определены.","retrieval":"Локальная библиотека: {count} проверенных совпадений.","research":"Поиск в открытом интернете: {count} источников.","planning":"Требования и справочные источники подготовлены.","generation":"Черновик кода получен.","quality_guard":"Бессвязная генерация отклонена.","repair_done":"Повторная генерация улучшила проверку.","repair_failed":"Повторная генерация не улучшила проверку."}
    },
    "en": {
        "plan":["Prepare a solution in {language}.","Compare references and prioritize official documentation.","Create a draft and run static checks without executing it."],
        "summary":{"retrieved_example":"A close verified example was found. Adapt it to your project before use.","model_not_loaded":"Research is complete, but SWIN weights are not loaded. No code was generated.","generation_failed":"Code generation could not be started.","generation_rejected":"Unstructured output was rejected instead of being presented as code. Clarify the task or use a verified example.","draft_validated":"A draft was created and passed the available static checks.","draft_needs_review":"A draft was created, but validation is limited or needs fixes."},
        "validation":{"passed":"Static validation passed.","limited":"Only limited static checks are available.","failed":"Static validation failed.","not_run":"Static validation was not run.","not_supported":"A full static checker is unavailable for this format."},
        "warnings":{"retrieved":"This is a similar reference example, not a fully adapted implementation.","adapt":"Check node paths, library versions and input data against your project.","limited":"A full compiler or parser is unavailable for this language.","failed":"The code did not pass static validation. Do not use it as a finished solution.","web":"Web sources may contain errors or outdated examples.","static":"Static checks do not prove that the algorithm is correct.","runtime":"The code and tests were not executed on the server.","deploy":"Deploy the server with a SWIN checkpoint to enable generation.","resources":"Check model loading and server resources."},
        "research_found":"Web references were found; this does not guarantee that every page is complete or accurate.","research_none":"No web references were retrieved.","runtime":"The code and tests were not run; the server does not execute untrusted programs.","cross_found":"Reference materials are attached for manual cross-checking.","cross_none":"No reference materials were retrieved for cross-checking.","empty":"Describe the coding task.","too_long":"The task description is too long (maximum 4,000 characters).",
        "activity":{"requirements":"Requirements identified.","retrieval":"Local library: {count} verified matches.","research":"Public web search: {count} sources.","planning":"Requirements and references are ready.","generation":"Code draft generated.","quality_guard":"Unstructured generation was rejected.","repair_done":"A retry improved the validation result.","repair_failed":"A retry did not improve the validation result."}
    },
    "de": {
        "plan":["Lösung in {language} vorbereiten.","Quellen vergleichen und offizielle Dokumentation bevorzugen.","Entwurf erstellen und statisch prüfen, ohne ihn auszuführen."],
        "summary":{"retrieved_example":"Ein passendes geprüftes Beispiel wurde gefunden. Passe es vor der Verwendung an dein Projekt an.","model_not_loaded":"Die Recherche ist abgeschlossen, aber die SWIN-Gewichte sind nicht geladen. Es wurde kein Code erzeugt.","generation_failed":"Die Codegenerierung konnte nicht gestartet werden.","generation_rejected":"Unstrukturierte Ausgabe wurde nicht als Code ausgegeben. Präzisiere die Aufgabe oder nutze ein geprüftes Beispiel.","draft_validated":"Ein Entwurf wurde erstellt und hat die verfügbaren statischen Prüfungen bestanden.","draft_needs_review":"Ein Entwurf wurde erstellt, aber die Prüfung ist eingeschränkt oder erfordert Korrekturen."},
        "validation":{"passed":"Statische Prüfung bestanden.","limited":"Es sind nur eingeschränkte statische Prüfungen verfügbar.","failed":"Statische Prüfung fehlgeschlagen.","not_run":"Statische Prüfung wurde nicht ausgeführt.","not_supported":"Für dieses Format ist kein vollständiger statischer Prüfer verfügbar."},
        "warnings":{"retrieved":"Dies ist ein ähnliches Referenzbeispiel und keine vollständig angepasste Implementierung.","adapt":"Prüfe Node-Pfade, Bibliotheksversionen und Eingabedaten in deinem Projekt.","limited":"Ein vollständiger Compiler oder Parser ist für diese Sprache nicht verfügbar.","failed":"Der Code hat die statische Prüfung nicht bestanden. Verwende ihn nicht als fertige Lösung.","web":"Webquellen können Fehler oder veraltete Beispiele enthalten.","static":"Statische Prüfungen beweisen nicht die algorithmische Korrektheit.","runtime":"Code und Tests wurden auf dem Server nicht ausgeführt.","deploy":"Stelle den Server mit einem SWIN-Checkpoint bereit, um die Generierung zu aktivieren.","resources":"Prüfe das Laden des Modells und die Serverressourcen."},
        "research_found":"Webquellen wurden gefunden; ihre Vollständigkeit und Richtigkeit ist damit nicht garantiert.","research_none":"Es wurden keine Webquellen abgerufen.","runtime":"Code und Tests wurden nicht ausgeführt; der Server führt keine nicht vertrauenswürdigen Programme aus.","cross_found":"Referenzmaterialien zur manuellen Prüfung sind beigefügt.","cross_none":"Es wurden keine Referenzmaterialien zur Gegenprüfung gefunden.","empty":"Beschreibe die Programmieraufgabe.","too_long":"Die Aufgabenbeschreibung ist zu lang (maximal 4.000 Zeichen).",
        "activity":{"requirements":"Anforderungen erfasst.","retrieval":"Lokale Bibliothek: {count} geprüfte Treffer.","research":"Öffentliche Websuche: {count} Quellen.","planning":"Anforderungen und Quellen vorbereitet.","generation":"Codeentwurf erstellt.","quality_guard":"Unstrukturierte Generierung verworfen.","repair_done":"Ein erneuter Versuch verbesserte das Prüfergebnis.","repair_failed":"Ein erneuter Versuch verbesserte das Prüfergebnis nicht."}
    },
    "es": {
        "plan":["Preparar una solución en {language}.","Comparar las referencias y priorizar la documentación oficial.","Crear un borrador y comprobarlo estáticamente sin ejecutarlo."],
        "summary":{"retrieved_example":"Se encontró un ejemplo verificado similar. Adáptalo a tu proyecto antes de usarlo.","model_not_loaded":"La búsqueda terminó, pero los pesos de SWIN no están cargados. No se generó código.","generation_failed":"No se pudo iniciar la generación de código.","generation_rejected":"Se rechazó el texto incoherente en lugar de mostrarlo como código. Aclara la tarea o usa un ejemplo verificado.","draft_validated":"Se creó un borrador y superó las comprobaciones estáticas disponibles.","draft_needs_review":"Se creó un borrador, pero la validación es limitada o requiere correcciones."},
        "validation":{"passed":"La validación estática se superó.","limited":"Solo hay comprobaciones estáticas limitadas.","failed":"La validación estática falló.","not_run":"No se ejecutó la validación estática.","not_supported":"No hay un verificador estático completo para este formato."},
        "warnings":{"retrieved":"Es un ejemplo de referencia similar, no una implementación completamente adaptada.","adapt":"Comprueba las rutas de nodos, las versiones de bibliotecas y los datos de entrada de tu proyecto.","limited":"No hay un compilador o analizador completo disponible para este lenguaje.","failed":"El código no superó la validación estática. No lo uses como solución terminada.","web":"Las fuentes web pueden contener errores o ejemplos obsoletos.","static":"Las comprobaciones estáticas no demuestran que el algoritmo sea correcto.","runtime":"El código y las pruebas no se ejecutaron en el servidor.","deploy":"Despliega el servidor con un checkpoint SWIN para habilitar la generación.","resources":"Comprueba la carga del modelo y los recursos del servidor."},
        "research_found":"Se encontraron fuentes web; esto no garantiza que todas las páginas sean completas o correctas.","research_none":"No se obtuvieron fuentes web.","runtime":"El código y las pruebas no se ejecutaron; el servidor no ejecuta programas no confiables.","cross_found":"Se adjuntan referencias para una comprobación manual.","cross_none":"No se obtuvieron referencias para contrastar.","empty":"Describe la tarea de programación.","too_long":"La descripción de la tarea es demasiado larga (máximo 4.000 caracteres).",
        "activity":{"requirements":"Requisitos identificados.","retrieval":"Biblioteca local: {count} coincidencias verificadas.","research":"Búsqueda web pública: {count} fuentes.","planning":"Requisitos y referencias preparados.","generation":"Borrador de código generado.","quality_guard":"Se rechazó la generación incoherente.","repair_done":"Un nuevo intento mejoró la validación.","repair_failed":"Un nuevo intento no mejoró la validación."}
    },
    "fr": {
        "plan":["Préparer une solution en {language}.","Comparer les références en privilégiant la documentation officielle.","Créer un brouillon et le vérifier statiquement sans l’exécuter."],
        "summary":{"retrieved_example":"Un exemple vérifié proche a été trouvé. Adapte-le à ton projet avant utilisation.","model_not_loaded":"La recherche est terminée, mais les poids SWIN ne sont pas chargés. Aucun code n’a été généré.","generation_failed":"La génération de code n’a pas pu démarrer.","generation_rejected":"Le texte incohérent a été rejeté au lieu d’être présenté comme du code. Précise la tâche ou utilise un exemple vérifié.","draft_validated":"Un brouillon a été créé et a réussi les vérifications statiques disponibles.","draft_needs_review":"Un brouillon a été créé, mais la validation est limitée ou nécessite des corrections."},
        "validation":{"passed":"Vérification statique réussie.","limited":"Seules des vérifications statiques limitées sont disponibles.","failed":"Échec de la vérification statique.","not_run":"La vérification statique n’a pas été exécutée.","not_supported":"Aucun vérificateur statique complet n’est disponible pour ce format."},
        "warnings":{"retrieved":"Il s’agit d’un exemple de référence similaire, pas d’une implémentation entièrement adaptée.","adapt":"Vérifie les chemins des nœuds, les versions des bibliothèques et les données d’entrée de ton projet.","limited":"Aucun compilateur ou analyseur complet n’est disponible pour ce langage.","failed":"Le code n’a pas réussi la vérification statique. Ne l’utilise pas comme solution finale.","web":"Les sources Web peuvent contenir des erreurs ou des exemples obsolètes.","static":"Les vérifications statiques ne prouvent pas l’exactitude de l’algorithme.","runtime":"Le code et les tests n’ont pas été exécutés sur le serveur.","deploy":"Déploie le serveur avec un checkpoint SWIN pour activer la génération.","resources":"Vérifie le chargement du modèle et les ressources du serveur."},
        "research_found":"Des sources Web ont été trouvées ; cela ne garantit pas leur exhaustivité ni leur exactitude.","research_none":"Aucune source Web n’a été récupérée.","runtime":"Le code et les tests n’ont pas été exécutés ; le serveur n’exécute pas de programmes non fiables.","cross_found":"Des références sont jointes pour une vérification manuelle.","cross_none":"Aucune référence n’a été obtenue pour la vérification.","empty":"Décris la tâche de programmation.","too_long":"La description de la tâche est trop longue (maximum 4 000 caractères).",
        "activity":{"requirements":"Exigences identifiées.","retrieval":"Bibliothèque locale : {count} correspondances vérifiées.","research":"Recherche Web publique : {count} sources.","planning":"Exigences et références préparées.","generation":"Brouillon de code généré.","quality_guard":"La génération incohérente a été rejetée.","repair_done":"Une nouvelle tentative a amélioré la validation.","repair_failed":"Une nouvelle tentative n’a pas amélioré la validation."}
    },
    "zh": {
        "plan":["准备 {language} 方案。","比较参考资料，并优先采用官方文档。","生成草稿并进行静态检查，不执行代码。"],
        "summary":{"retrieved_example":"找到了相近的已验证示例。使用前请根据你的项目进行调整。","model_not_loaded":"资料检索已完成，但 SWIN 权重尚未加载，因此没有生成代码。","generation_failed":"无法启动代码生成。","generation_rejected":"生成内容不连贯，已拒绝将其作为代码展示。请补充任务要求或使用已验证示例。","draft_validated":"已生成草稿，并通过当前可用的静态检查。","draft_needs_review":"已生成草稿，但验证能力有限或仍需修正。"},
        "validation":{"passed":"静态检查通过。","limited":"目前只能进行有限的静态检查。","failed":"静态检查未通过。","not_run":"尚未执行静态检查。","not_supported":"此格式没有完整的静态检查器。"},
        "warnings":{"retrieved":"这是相近的参考示例，并非完全适配你项目的实现。","adapt":"请核对节点路径、库版本以及项目输入数据。","limited":"此语言目前没有可用的完整编译器或解析器。","failed":"代码未通过静态检查，请勿将其作为最终方案使用。","web":"网页资料可能包含错误或过时示例。","static":"静态检查无法证明算法逻辑正确。","runtime":"服务器没有执行代码和测试。","deploy":"部署包含 SWIN checkpoint 的服务器后才能启用代码生成。","resources":"请检查模型加载状态和服务器资源。"},
        "research_found":"已找到网页资料，但这不保证每个页面都完整或准确。","research_none":"没有获取到网页资料。","runtime":"代码和测试未运行；服务器不会执行不可信程序。","cross_found":"已附上参考资料，供手动核对。","cross_none":"没有获取到可供核对的参考资料。","empty":"请描述编程任务。","too_long":"任务描述过长（最多 4000 个字符）。",
        "activity":{"requirements":"已识别需求。","retrieval":"本地资料库：{count} 个已验证匹配项。","research":"公开网页搜索：{count} 个来源。","planning":"需求和参考资料已准备好。","generation":"代码草稿已生成。","quality_guard":"已拒绝不连贯的生成结果。","repair_done":"重新生成后验证结果有所改善。","repair_failed":"重新生成未能改善验证结果。"}
    },
    "uk": {
        "plan":["Підготувати рішення мовою {language}.","Зіставити матеріали та надати пріоритет офіційній документації.","Створити чернетку й статично перевірити її без запуску."],
        "summary":{"retrieved_example":"Знайдено близький перевірений приклад. Перед використанням адаптуй його до проєкту.","model_not_loaded":"Пошук завершено, але ваги SWIN не завантажені. Код не генерувався.","generation_failed":"Не вдалося запустити генерацію коду.","generation_rejected":"Незв’язний текст відхилено замість показу як коду. Уточни завдання або використай перевірений приклад.","draft_validated":"Чернетку створено й перевірено доступними статичними засобами.","draft_needs_review":"Чернетку створено, але перевірка обмежена або потрібні виправлення."},
        "validation":{"passed":"Статичну перевірку пройдено.","limited":"Доступні лише обмежені статичні перевірки.","failed":"Статичну перевірку не пройдено.","not_run":"Статичну перевірку не виконано.","not_supported":"Для цього формату немає повноцінного статичного перевіряльника."},
        "warnings":{"retrieved":"Це схожий приклад, а не повністю адаптована реалізація.","adapt":"Перевір шляхи вузлів, версії бібліотек і вхідні дані свого проєкту.","limited":"Для цієї мови немає повноцінного компілятора або парсера.","failed":"Код не пройшов статичну перевірку. Не використовуй його як готове рішення.","web":"Вебджерела можуть містити помилки або застарілі приклади.","static":"Статична перевірка не доводить правильність алгоритму.","runtime":"Код і тести не запускалися на сервері.","deploy":"Розгорни сервер із checkpoint SWIN, щоб увімкнути генерацію.","resources":"Перевір завантаження моделі та ресурси сервера."},
        "research_found":"Знайдено вебджерела, але це не гарантує повноти чи точності кожної сторінки.","research_none":"Вебджерела не отримано.","runtime":"Код і тести не запускалися: сервер не виконує недовірені програми.","cross_found":"До результату додано матеріали для ручної перевірки.","cross_none":"Матеріалів для звіряння не отримано.","empty":"Опиши завдання для програмування.","too_long":"Опис завдання задовгий (максимум 4000 символів).",
        "activity":{"requirements":"Вимоги визначено.","retrieval":"Локальна бібліотека: {count} перевірених збігів.","research":"Пошук у відкритому інтернеті: {count} джерел.","planning":"Вимоги та довідкові матеріали підготовлено.","generation":"Чернетку коду отримано.","quality_guard":"Незв’язну генерацію відхилено.","repair_done":"Повторна генерація покращила результат перевірки.","repair_failed":"Повторна генерація не покращила результат перевірки."}
    },
    "pt": {
        "plan":["Preparar uma solução em {language}.","Comparar as referências e priorizar a documentação oficial.","Criar um rascunho e verificá-lo estaticamente sem executá-lo."],
        "summary":{"retrieved_example":"Foi encontrado um exemplo verificado semelhante. Adapte-o ao seu projeto antes de usar.","model_not_loaded":"A pesquisa terminou, mas os pesos SWIN não estão carregados. Nenhum código foi gerado.","generation_failed":"Não foi possível iniciar a geração de código.","generation_rejected":"O texto incoerente foi rejeitado em vez de ser apresentado como código. Esclareça a tarefa ou use um exemplo verificado.","draft_validated":"Um rascunho foi criado e passou pelas verificações estáticas disponíveis.","draft_needs_review":"Um rascunho foi criado, mas a validação é limitada ou requer correções."},
        "validation":{"passed":"Verificação estática aprovada.","limited":"Apenas verificações estáticas limitadas estão disponíveis.","failed":"A verificação estática falhou.","not_run":"A verificação estática não foi executada.","not_supported":"Não há um verificador estático completo para este formato."},
        "warnings":{"retrieved":"Este é um exemplo de referência semelhante, não uma implementação totalmente adaptada.","adapt":"Confira os caminhos dos nós, versões das bibliotecas e dados de entrada do seu projeto.","limited":"Não há um compilador ou analisador completo disponível para esta linguagem.","failed":"O código não passou na verificação estática. Não o use como solução final.","web":"Fontes da web podem conter erros ou exemplos desatualizados.","static":"Verificações estáticas não comprovam a correção do algoritmo.","runtime":"O código e os testes não foram executados no servidor.","deploy":"Implante o servidor com um checkpoint SWIN para habilitar a geração.","resources":"Verifique o carregamento do modelo e os recursos do servidor."},
        "research_found":"Foram encontradas fontes na web; isso não garante que todas as páginas sejam completas ou corretas.","research_none":"Nenhuma fonte da web foi obtida.","runtime":"O código e os testes não foram executados; o servidor não executa programas não confiáveis.","cross_found":"Referências foram anexadas para conferência manual.","cross_none":"Nenhuma referência foi obtida para conferência.","empty":"Descreva a tarefa de programação.","too_long":"A descrição da tarefa é muito longa (máximo de 4.000 caracteres).",
        "activity":{"requirements":"Requisitos identificados.","retrieval":"Biblioteca local: {count} correspondências verificadas.","research":"Pesquisa pública na web: {count} fontes.","planning":"Requisitos e referências preparados.","generation":"Rascunho de código gerado.","quality_guard":"A geração incoerente foi rejeitada.","repair_done":"Uma nova tentativa melhorou a validação.","repair_failed":"Uma nova tentativa não melhorou a validação."}
    },
    "it": {
        "plan":["Preparare una soluzione in {language}.","Confrontare le fonti dando priorità alla documentazione ufficiale.","Creare una bozza e verificarla staticamente senza eseguirla."],
        "summary":{"retrieved_example":"È stato trovato un esempio verificato simile. Adattalo al progetto prima di usarlo.","model_not_loaded":"La ricerca è terminata, ma i pesi SWIN non sono caricati. Non è stato generato codice.","generation_failed":"Impossibile avviare la generazione del codice.","generation_rejected":"Il testo incoerente è stato rifiutato e non mostrato come codice. Chiarisci il compito o usa un esempio verificato.","draft_validated":"È stata creata una bozza che ha superato i controlli statici disponibili.","draft_needs_review":"È stata creata una bozza, ma la convalida è limitata o richiede correzioni."},
        "validation":{"passed":"Controllo statico superato.","limited":"Sono disponibili solo controlli statici limitati.","failed":"Controllo statico non superato.","not_run":"Il controllo statico non è stato eseguito.","not_supported":"Non è disponibile un controllo statico completo per questo formato."},
        "warnings":{"retrieved":"Questo è un esempio di riferimento simile, non un'implementazione completamente adattata.","adapt":"Verifica percorsi dei nodi, versioni delle librerie e dati di input del progetto.","limited":"Per questo linguaggio non è disponibile un compilatore o parser completo.","failed":"Il codice non ha superato il controllo statico. Non usarlo come soluzione definitiva.","web":"Le fonti Web possono contenere errori o esempi non aggiornati.","static":"I controlli statici non dimostrano la correttezza dell'algoritmo.","runtime":"Codice e test non sono stati eseguiti sul server.","deploy":"Distribuisci il server con un checkpoint SWIN per abilitare la generazione.","resources":"Controlla il caricamento del modello e le risorse del server."},
        "research_found":"Sono state trovate fonti Web; ciò non garantisce completezza o accuratezza di ogni pagina.","research_none":"Nessuna fonte Web è stata recuperata.","runtime":"Codice e test non sono stati eseguiti: il server non esegue programmi non attendibili.","cross_found":"Sono allegati riferimenti per un controllo manuale.","cross_none":"Non sono stati recuperati riferimenti per il confronto.","empty":"Descrivi il compito di programmazione.","too_long":"La descrizione del compito è troppo lunga (massimo 4.000 caratteri).",
        "activity":{"requirements":"Requisiti identificati.","retrieval":"Libreria locale: {count} corrispondenze verificate.","research":"Ricerca Web pubblica: {count} fonti.","planning":"Requisiti e riferimenti pronti.","generation":"Bozza di codice generata.","quality_guard":"Generazione incoerente rifiutata.","repair_done":"Un nuovo tentativo ha migliorato la convalida.","repair_failed":"Un nuovo tentativo non ha migliorato la convalida."}
    },
    "ja": {
        "plan":["{language} で解決策を準備します。","公式ドキュメントを優先して参考資料を比較します。","コードを実行せずに下書きを作成し、静的検査を行います。"],
        "summary":{"retrieved_example":"近い内容の検証済みサンプルが見つかりました。使用前にプロジェクトに合わせて調整してください。","model_not_loaded":"検索は完了しましたが、SWIN の重みが読み込まれていないためコードは生成されませんでした。","generation_failed":"コード生成を開始できませんでした。","generation_rejected":"一貫性のない出力をコードとして表示せずに拒否しました。タスクを詳しくするか、検証済みサンプルを使ってください。","draft_validated":"下書きを作成し、利用可能な静的検査に合格しました。","draft_needs_review":"下書きを作成しましたが、検証は限定的か修正が必要です。"},
        "validation":{"passed":"静的検査に合格しました。","limited":"限定的な静的検査のみ利用できます。","failed":"静的検査に失敗しました。","not_run":"静的検査は実行されていません。","not_supported":"この形式には完全な静的検査機能がありません。"},
        "warnings":{"retrieved":"これは類似した参考例であり、プロジェクトに完全適合した実装ではありません。","adapt":"ノードのパス、ライブラリのバージョン、入力データを確認してください。","limited":"この言語向けの完全なコンパイラまたはパーサーは利用できません。","failed":"コードは静的検査に合格していません。完成した解決策として使用しないでください。","web":"Web の情報には誤りや古い例が含まれる可能性があります。","static":"静的検査だけではアルゴリズムの正しさを証明できません。","runtime":"サーバー上でコードやテストは実行されていません。","deploy":"コード生成には SWIN checkpoint を含むサーバーのデプロイが必要です。","resources":"モデルの読み込み状態とサーバーのリソースを確認してください。"},
        "research_found":"Web 参照情報が見つかりましたが、各ページの完全性や正確性を保証するものではありません。","research_none":"Web 参照情報を取得できませんでした。","runtime":"コードとテストは実行されていません。サーバーは信頼できないプログラムを実行しません。","cross_found":"手動確認用の参考資料を添付しました。","cross_none":"照合用の参考資料は取得できませんでした。","empty":"プログラミングのタスクを説明してください。","too_long":"タスクの説明が長すぎます（最大 4000 文字）。",
        "activity":{"requirements":"要件を確認しました。","retrieval":"ローカルライブラリ: 検証済み候補 {count} 件。","research":"公開 Web 検索: {count} 件。","planning":"要件と参考資料を準備しました。","generation":"コードの下書きを生成しました。","quality_guard":"一貫性のない生成結果を拒否しました。","repair_done":"再生成によって検証結果が改善しました。","repair_failed":"再生成しても検証結果は改善しませんでした。"}
    },
}


def code_assist_copy(language: str) -> dict:
    normalized = normalize_response_language(language)
    if normalized == "auto":
        normalized = "ru"
    return CODE_ASSIST_COPY[normalized]
