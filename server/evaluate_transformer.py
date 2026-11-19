"""Acceptance tests for generated SWIN text and code before publishing checkpoints."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from transformer import load_transformer
from language_support import format_training_prompt


def assess_generated_text(
    text: str,
    require_code: bool = False,
    required_concepts: list[list[str]] | None = None,
    required_function: str | None = None,
) -> dict[str, Any]:
    value = (text or "").strip()
    reasons: list[str] = []
    if len(value) < (8 if require_code else 24):
        reasons.append("слишком короткий ответ")
    if "\ufffd" in value:
        reasons.append("некорректный UTF-8")
    if any(ord(char) < 32 and char not in "\n\t\r" for char in value):
        reasons.append("управляющие символы в ответе")

    letters = [char.lower() for char in value if char.isalpha()]
    if len(letters) < 6:
        reasons.append("слишком мало букв")
    elif len(set(letters)) / len(letters) < 0.14:
        reasons.append("низкое разнообразие букв")
    if letters:
        counts: dict[str, int] = {}
        for char in letters:
            counts[char] = counts.get(char, 0) + 1
        if max(counts.values()) / len(letters) > 0.48:
            reasons.append("один символ повторяется слишком часто")
        if re.search(r"(.)\1{5,}", "".join(letters)):
            reasons.append("длинная серия повторяющихся символов")

    if require_code:
        code = value
        fence = chr(96) * 3
        match = re.search(
            re.escape(fence) + r"(?:python)?\s*\n(.*?)" + re.escape(fence),
            value,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            code = match.group(1).strip()
        try:
            ast.parse(code)
        except SyntaxError as error:
            reasons.append(f"синтаксис Python не прошёл AST: {error.msg}")
        if not re.search(r"\b(def|class)\b", code):
            reasons.append("не найдено определение функции или класса")
        if required_function:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                tree = None
            if tree is not None:
                function = next(
                    (
                        node for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == required_function
                    ),
                    None,
                )
                if function is None:
                    reasons.append(f"не найдена обязательная функция {required_function}")
                else:
                    argument_names = {argument.arg for argument in function.args.args}
                    if not {"a", "b"}.issubset(argument_names):
                        reasons.append("функция add должна принимать аргументы a и b")
                    has_add_return = any(
                        isinstance(node, ast.Return)
                        and isinstance(node.value, ast.BinOp)
                        and isinstance(node.value.op, ast.Add)
                        and isinstance(node.value.left, ast.Name)
                        and isinstance(node.value.right, ast.Name)
                        and {node.value.left.id, node.value.right.id} == {"a", "b"}
                        for node in ast.walk(function)
                    )
                    if not has_add_return:
                        reasons.append("функция add должна возвращать сумму a и b")
    else:
        has_cjk = any(
            "\u4e00" <= char <= "\u9fff"
            or "\u3040" <= char <= "\u30ff"
            for char in value
        )
        words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", value)
        if len(words) < 4 and not has_cjk:
            reasons.append("недостаточно слов для связного ответа")
        if has_cjk and len(letters) < 10:
            reasons.append("слишком короткий ответ на CJK-языке")
        if required_concepts:
            lowered = value.casefold()
            missing = [
                "/".join(options)
                for options in required_concepts
                if not any(option.casefold() in lowered for option in options)
            ]
            if missing:
                reasons.append("не раскрыты ключевые понятия: " + ", ".join(missing))

    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "preview": value[:220],
    }


def evaluate_model(model) -> dict[str, Any]:
    """Evaluate Russian plus every curated chat language and a deterministic Python task."""
    text_probes = [
        {
            "name": "russian_explanation",
            "language": "ru",
            "question": "Объясни простыми словами, что такое функция в Python.",
            "required_concepts": [["функц"], ["код"], ["результат", "задач"]],
        },
        {
            "name": "russian_debugging",
            "language": "ru",
            "question": "Как по шагам найти причину ошибки TypeError в Python?",
            "required_concepts": [["ошиб"], ["тип", "значен"], ["строк", "traceback"]],
        },
        {
            "name": "english_explanation",
            "language": "en",
            "question": "Explain briefly what a function in Python is.",
            "required_concepts": [["function"], ["code"], ["return", "result"]],
        },
        {
            "name": "german_explanation",
            "language": "de",
            "question": "Erkläre kurz, was eine Funktion in Python ist.",
            "required_concepts": [["funktion"], ["code"], ["ergebnis", "rückgabe", "zurück"]],
        },
        {
            "name": "spanish_explanation",
            "language": "es",
            "question": "Explica brevemente qué es una función en Python.",
            "required_concepts": [["función", "funcion"], ["código", "codigo"], ["resultado", "devolver"]],
        },
        {
            "name": "french_explanation",
            "language": "fr",
            "question": "Explique brièvement ce qu'est une fonction en Python.",
            "required_concepts": [["fonction"], ["code"], ["résultat", "resultat", "renvoie"]],
        },
        {
            "name": "chinese_explanation",
            "language": "zh",
            "question": "请用简单的话解释 Python 中的函数是什么。",
            "required_concepts": [["函数"], ["代码"], ["结果", "返回"]],
        },
        {
            "name": "ukrainian_explanation",
            "language": "uk",
            "question": "Поясни простими словами, що таке функція в Python.",
            "required_concepts": [["функц"], ["код"], ["результат", "поверта"]],
        },
        {
            "name": "portuguese_explanation",
            "language": "pt",
            "question": "Explique brevemente o que é uma função em Python.",
            "required_concepts": [["função", "funcao"], ["código", "codigo"], ["resultado", "retorna"]],
        },
        {
            "name": "italian_explanation",
            "language": "it",
            "question": "Spiega brevemente cos'è una funzione in Python.",
            "required_concepts": [["funzione"], ["codice"], ["risultato", "restituisce"]],
        },
        {
            "name": "japanese_explanation",
            "language": "ja",
            "question": "Pythonの関数とは何か、簡単に説明してください。",
            "required_concepts": [["関数"], ["コード"], ["結果", "返"]],
        },
    ]

    results = []
    for probe in text_probes:
        try:
            prompt = format_training_prompt(probe["question"], probe["language"])
            generated = model.generate(
                prompt,
                max_new_tokens=180,
                temperature=0.0,
                top_k=1,
                stop_at_newline=False,
            )
            result = assess_generated_text(
                generated,
                required_concepts=probe["required_concepts"],
            )
        except Exception as error:
            result = {
                "status": "failed",
                "reasons": [f"ошибка генерации: {str(error)[:180]}"],
                "preview": "",
            }
        results.append({
            "name": probe["name"],
            "language": probe["language"],
            **result,
        })

    code_probe = {
        "name": "python_function",
        "language": "en",
    }
    try:
        code_prompt = format_training_prompt(
            "Write a Python function add(a, b) that returns the sum of two numbers. Return the complete code.",
            "en",
        )
        generated_code = model.generate(
            code_prompt,
            max_new_tokens=180,
            temperature=0.0,
            top_k=1,
            stop_at_newline=False,
        )
        code_result = assess_generated_text(
            generated_code,
            require_code=True,
            required_function="add",
        )
    except Exception as error:
        code_result = {
            "status": "failed",
            "reasons": [f"ошибка генерации: {str(error)[:180]}"],
            "preview": "",
        }
    results.append({**code_probe, **code_result})

    passed_text = sum(
        item["status"] == "passed"
        for item in results
        if item["name"] != "python_function"
    )
    total_text = len(text_probes)
    code_passed = results[-1]["status"] == "passed"
    passed_languages = [
        item["language"] for item in results
        if item["name"] != "python_function" and item["status"] == "passed"
    ]
    failed_languages = [
        item["language"] for item in results
        if item["name"] != "python_function" and item["status"] != "passed"
    ]
    accepted = passed_text == total_text and code_passed
    return {
        "accepted": accepted,
        "passed_text_probes": passed_text,
        "total_text_probes": total_text,
        "passed_languages": passed_languages,
        "failed_languages": failed_languages,
        "python_code_probe_passed": code_passed,
        "probes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SWIN checkpoint quality gate")
    parser.add_argument("--checkpoint", default="models/swin-transformer.pt")
    parser.add_argument("--json-output", default="models/evaluation.json")
    args = parser.parse_args()

    model = load_transformer(args.checkpoint)
    report = evaluate_model(model)
    output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        print("QUALITY GATE FAILED: checkpoint will not be published.")
        return 1
    print("QUALITY GATE PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
