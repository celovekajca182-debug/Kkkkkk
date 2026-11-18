"""Research-first coding assistant for SWIN AI.

Only concise task summaries, public documentation sources and validation results are
returned. Private chain-of-thought is never exposed.
"""
from __future__ import annotations

import ast
import re
from typing import Any

LANGUAGES = {
    "auto": "автоматически",
    "python": "Python",
    "gdscript": "Godot 4 GDScript",
    "kotlin": "Kotlin",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "html": "HTML",
    "css": "CSS",
    "json": "JSON",
}

LANGUAGE_HINTS = {
    "python": ("python", "питон", "fastapi", "django", "flask"),
    "gdscript": ("godot", "gdscript", "гдскрипт"),
    "kotlin": ("kotlin", "android", "андроид"),
    "javascript": ("javascript", "js", "node", "react"),
    "typescript": ("typescript", "ts"),
    "html": ("html", "веб-страниц", "сайт"),
    "css": ("css", "стил"),
}


def detect_language(task: str, requested: str) -> str:
    normalized = (requested or "auto").strip().lower()
    if normalized in LANGUAGES and normalized != "auto":
        return normalized
    lowered = task.lower()
    for language, hints in LANGUAGE_HINTS.items():
        if any(hint in lowered for hint in hints):
            return language
    return "python"


def _search_query(task: str, language: str) -> str:
    docs = {
        "python": "site:docs.python.org/3/",
        "gdscript": "site:docs.godotengine.org/en/stable/",
        "kotlin": "site:kotlinlang.org/docs/ OR site:developer.android.com/",
        "javascript": "site:developer.mozilla.org/",
        "typescript": "site:typescriptlang.org/docs/",
        "html": "site:developer.mozilla.org/",
        "css": "site:developer.mozilla.org/",
    }.get(language, "")
    compact_task = re.sub(r"\s+", " ", task).strip()[:220]
    return " ".join(part for part in (
        compact_task, LANGUAGES.get(language, language), docs, "official documentation API examples"
    ) if part)


def search_documentation(task: str, language: str, max_results: int = 4) -> list[dict[str, str]]:
    """Search public web snippets; no AI service or paid model API is used."""
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    try:
        raw_results = DDGS(timeout=8).text(
            _search_query(task, language),
            region="ru-ru",
            safesearch="moderate",
            max_results=max(1, min(max_results, 6)),
            backend="auto",
        )
    except Exception:
        return []

    results: list[dict[str, str]] = []
    for item in raw_results or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("href", "")).strip()
        if not url.startswith("https://"):
            continue
        results.append({
            "title": str(item.get("title", "Источник"))[:180],
            "url": url[:600],
            "snippet": str(item.get("body", ""))[:700],
        })
    return results[:max_results]


def _extract_code(text: str) -> str:
    fence = chr(96) * 3
    pattern = re.escape(fence) + r"(?:[a-zA-Z0-9_+.-]+)?\s*\n(.*?)" + re.escape(fence)
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip().strip(chr(96)).strip()


def validate_code(code: str, language: str) -> dict[str, Any]:
    """Run safe static parsers only. Generated code is never executed on the API server."""
    if not code:
        return {"status": "failed", "message": "Модель не вернула код."}

    if language == "python":
        try:
            ast.parse(code)
            return {
                "status": "passed",
                "method": "python_ast",
                "message": "Синтаксис Python корректен. Программа не запускалась.",
            }
        except SyntaxError as error:
            return {
                "status": "failed",
                "method": "python_ast",
                "message": f"Ошибка Python в строке {error.lineno}: {error.msg}",
            }

    if language == "json":
        import json
        try:
            json.loads(code)
            return {"status": "passed", "method": "json_parser", "message": "JSON корректен."}
        except json.JSONDecodeError as error:
            return {
                "status": "failed",
                "method": "json_parser",
                "message": f"Ошибка JSON в строке {error.lineno}: {error.msg}",
            }

    if language in {"gdscript", "kotlin", "javascript", "typescript", "html", "css"}:
        pairs = {"(": ")", "[": "]", "{": "}"}
        closing = {value: key for key, value in pairs.items()}
        stack = []
        quote = None
        escaped = False
        for char in code:
            if quote:
                if escaped:
                    escaped = False
                elif char == chr(92):
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
            elif char in pairs:
                stack.append(char)
            elif char in closing:
                if not stack or stack.pop() != closing[char]:
                    return {
                        "status": "failed",
                        "method": "delimiter_heuristic",
                        "message": "Не совпадают скобки в черновике.",
                    }
        if quote or stack:
            return {
                "status": "failed",
                "method": "delimiter_heuristic",
                "message": "Возможно, не закрыта строка или скобка.",
            }
        return {
            "status": "limited",
            "method": "delimiter_heuristic",
            "message": "Базовая проверка скобок пройдена. Нужна проверка настоящим компилятором/движком.",
        }

    return {
        "status": "not_supported",
        "method": "none",
        "message": "Для этого языка в сервере пока нет статического парсера.",
    }


def _make_prompt(task: str, language: str, sources: list[dict[str, str]],
                 project_context: str, block_size: int) -> str:
    source_lines = []
    for source in sources[:3]:
        snippet = re.sub(r"\s+", " ", source.get("snippet", "")).strip()
        if snippet:
            source_lines.append(snippet[:150])

    prefix = (
        "Ты — SWIN, опытный программист. Сгенерируй рабочий законченный код. "
        "Не пиши размышления. Верни только код, без вступления. "
        f"Язык: {LANGUAGES.get(language, language)}. "
    )
    docs = " ".join(source_lines)
    context = re.sub(r"\s+", " ", project_context).strip()[:180]
    prompt = (
        f"{prefix}Контекст проекта: {context}. "
        f"Справка: {docs}. "
        f"ЗАДАЧА: {task.strip()} КОД:"
    )
    byte_limit = max(64, block_size - 4)
    encoded = prompt.encode("utf-8")
    if len(encoded) > byte_limit:
        prompt = encoded[-byte_limit:].decode("utf-8", errors="ignore")
    return prompt


def run_code_assist(
    task: str,
    language: str,
    transformer_model,
    research_web: bool = True,
    project_context: str = "",
    show_activity: bool = False,
) -> dict[str, Any]:
    """Research, plan, generate and statically validate one code draft."""
    normalized_task = (task or "").strip()
    if not normalized_task:
        raise ValueError("Опиши задачу, для которой нужен код.")
    if len(normalized_task) > 4000:
        raise ValueError("Описание задачи слишком длинное (максимум 4000 символов).")

    language = detect_language(normalized_task, language)
    activity = [
        {"stage": "requirements", "status": "done", "label": "Требования определены"},
    ]
    sources = search_documentation(normalized_task, language) if research_web else []
    activity.append({
        "stage": "research",
        "status": "done" if sources else ("skipped" if not research_web else "unavailable"),
        "label": "Найдено источников: " + str(len(sources)),
    })
    plan = [
        f"Подготовить решение на {LANGUAGES.get(language, language)}.",
        "Сверить используемые API с документацией, если источники доступны.",
        "Сформировать код и проверить его структуру.",
    ]

    if transformer_model is None:
        result = {
            "status": "model_not_loaded",
            "language": language,
            "summary": "Поиск и планирование выполнены, но веса SWIN не загружены на сервер. Код не был сгенерирован.",
            "code": "",
            "validation": {
                "status": "not_run",
                "message": "Для генерации требуется SWIN Transformer checkpoint.",
            },
            "plan": plan,
            "references": sources,
            "warnings": ["Разверни сервер с актуальным models/swin-transformer.pt."],
        }
        if show_activity:
            result["activity"] = activity
        return result

    activity.append({"stage": "planning", "status": "done", "label": "План реализации подготовлен"})
    block_size = int(getattr(transformer_model, "block_size", 256))
    prompt = _make_prompt(
        normalized_task, language, sources, project_context, block_size
    )
    try:
        raw = transformer_model.generate(
            prompt,
            max_new_tokens=900,
            temperature=0.12,
            top_k=12,
            stop_at_newline=False,
        )
    except Exception as error:
        result = {
            "status": "generation_failed",
            "language": language,
            "summary": "Не удалось запустить генерацию кода.",
            "code": "",
            "validation": {"status": "not_run", "message": str(error)[:300]},
            "plan": plan,
            "references": sources,
            "warnings": ["Проверь модель и серверные ресурсы."],
        }
        if show_activity:
            result["activity"] = activity
        return result

    activity.append({"stage": "generation", "status": "done", "label": "Черновик кода получен"})
    code = _extract_code(raw)
    validation = validate_code(code, language)

    if language == "python" and validation["status"] == "failed":
        repair_prompt = (
            f"Исправь синтаксис Python. Ошибка: {validation['message']}. "
            f"Верни только полный исправленный код: {code[:500]}"
        )
        repair_prompt_bytes = repair_prompt.encode("utf-8")
        repair_prompt = repair_prompt_bytes[-max(64, block_size - 4):].decode("utf-8", "ignore")
        try:
            repaired = transformer_model.generate(
                repair_prompt,
                max_new_tokens=900,
                temperature=0.0,
                top_k=1,
                stop_at_newline=False,
            )
            repaired_code = _extract_code(repaired)
            repaired_validation = validate_code(repaired_code, language)
            if repaired_validation["status"] == "passed":
                code, validation = repaired_code, repaired_validation
                activity.append({
                    "stage": "repair",
                    "status": "done",
                    "label": "Синтаксическая ошибка исправлена повторной генерацией",
                })
            else:
                activity.append({
                    "stage": "repair",
                    "status": "failed",
                    "label": "Повторная генерация не прошла проверку",
                })
        except Exception:
            activity.append({
                "stage": "repair",
                "status": "failed",
                "label": "Не удалось повторно сгенерировать код",
            })

    activity.append({
        "stage": "validation",
        "status": "done" if validation["status"] == "passed" else validation["status"],
        "label": validation["message"],
    })
    warnings = []
    if validation["status"] == "limited":
        warnings.append("Проверка ограниченная: проектный компилятор/движок не запускался.")
    elif validation["status"] == "not_supported":
        warnings.append("Нужна ручная проверка или отдельный компилятор.")
    elif validation["status"] == "failed":
        warnings.append("Код не прошёл статическую проверку; не считай его готовым.")
    warnings.append("SWIN не запускает сгенерированный код на сервере.")
    warnings.append("Наличие валидного синтаксиса не гарантирует правильность логики.")

    status = "empty_generation" if not code else (
        "draft_validated" if validation["status"] == "passed" else "draft_needs_review"
    )
    result = {
        "status": status,
        "language": language,
        "summary": (
            "Черновик кода создан и прошёл статическую проверку."
            if validation["status"] == "passed"
            else "Черновик создан; проверь его перед использованием."
        ),
        "code": code,
        "validation": validation,
        "plan": plan,
        "references": sources,
        "warnings": warnings,
    }
    if show_activity:
        result["activity"] = activity
    return result
