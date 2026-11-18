"""Prepare a small, license-attributed Python code-generation set from MBPP.

Uses only the sanitized human-reviewed configuration and preserves the official
train/evaluation split. It does not execute dataset code or its tests.
"""
import argparse
import ast
import json
from pathlib import Path


def valid_example(row: dict) -> dict | None:
    question = str(row.get("prompt") or row.get("text") or "").strip()
    code = str(row.get("code", "")).strip()
    tests = row.get("test_list") or []
    if not question or not code:
        return None
    if len(question) > 1200 or len(code) > 5000:
        return None
    try:
        ast.parse(code)
    except SyntaxError:
        return None

    test_lines = [str(test).strip() for test in tests if str(test).strip()]
    # The task and its expected behavior form the prompt; the model learns the code as answer.
    prompt = "Задача Python: " + question
    if test_lines:
        prompt += "\nПроверочные примеры:\n" + "\n".join(test_lines[:3])
    return {
        "question": prompt,
        "answer": code,
        "task_id": row.get("task_id"),
    }


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare sanitized MBPP train/evaluation code examples.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-train", type=int, default=350)
    parser.add_argument("--max-validation", type=int, default=90)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Установи datasets для загрузки MBPP.") from error

    dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
    if "train" not in dataset or "evaluation" not in dataset:
        raise RuntimeError(
            "В MBPP sanitized ожидались официальные split train и evaluation; "
            f"получены: {list(dataset.keys())}"
        )

    train_rows = []
    for row in dataset["train"]:
        item = valid_example(row)
        if item:
            train_rows.append(item)
        if len(train_rows) >= args.max_train:
            break

    validation_rows = []
    for row in dataset["evaluation"]:
        item = valid_example(row)
        if item:
            validation_rows.append(item)
        if len(validation_rows) >= args.max_validation:
            break

    if len(train_rows) < 20 or len(validation_rows) < 5:
        raise RuntimeError(
            "Слишком мало пригодных задач MBPP после AST-фильтрации: "
            f"train={len(train_rows)}, validation={len(validation_rows)}"
        )

    output = Path(args.output_dir)
    write_jsonl(output / "mbpp_python_train.jsonl", train_rows)
    write_jsonl(output / "mbpp_python_validation.jsonl", validation_rows)

    manifest = {
        "dataset": "google-research-datasets/mbpp",
        "configuration": "sanitized",
        "license": "CC-BY-4.0",
        "source": "https://huggingface.co/datasets/google-research-datasets/mbpp",
        "paper": "https://arxiv.org/abs/2108.07732",
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "filter": "AST parse only; no example code or test code was executed.",
        "notice": (
            "MBPP is an English/Python benchmark with around 1,000 human-written tasks. "
            "It improves only the initial Python code-training coverage, not every language."
        ),
    }
    (output / "mbpp_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
