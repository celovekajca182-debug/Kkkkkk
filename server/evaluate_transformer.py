"""Acceptance tests for generated SWIN text and code before publishing checkpoints."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from transformer import load_transformer


def assess_generated_text(text: str, require_code: bool = False) -> dict[str, Any]:
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
    elif len(set(letters)) / len(letters) < 0.22:
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
    else:
        words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", value)
        if len(words) < 4:
            reasons.append("недостаточно слов для связного ответа")

    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "preview": value[:220],
    }


def evaluate_model(model) -> dict[str, Any]:
    """Evaluate fixed prompts; no training samples are reused as expected answers."""
    probes = [
        {
            "name": "russian_explanation",
            "prompt": "Пользователь: Объясни простыми словами, что такое функция в Python.\\nSWIN:",
            "require_code": False,
        },
        {
            "name": "russian_debugging",
            "prompt": "Пользователь: Как по шагам найти причину ошибки TypeError в Python?\\nSWIN:",
            "require_code": False,
        },
        {
            "name": "python_function",
            "prompt": "Пользователь: Напиши функцию Python add(a, b), которая возвращает сумму двух чисел. Верни полный код.\\nSWIN:",
            "require_code": True,
        },
    ]
    results = []
    for probe in probes:
        try:
            generated = model.generate(
                probe["prompt"],
                max_new_tokens=180,
                temperature=0.0,
                top_k=1,
                stop_at_newline=False,
            )
            result = assess_generated_text(generated, require_code=probe["require_code"])
        except Exception as error:
            result = {
                "status": "failed",
                "reasons": [f"ошибка генерации: {str(error)[:180]}"],
                "preview": "",
            }
        results.append({"name": probe["name"], **result})

    passed_text = sum(
        item["status"] == "passed"
        for item in results
        if item["name"] != "python_function"
    )
    code_passed = next(
        item["status"] == "passed"
        for item in results
        if item["name"] == "python_function"
    )
    accepted = passed_text == 2 and code_passed
    return {
        "accepted": accepted,
        "passed_text_probes": passed_text,
        "total_text_probes": 2,
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
