"""Сбор небольшого проверяемого русскоязычного корпуса из открытых датасетов.

Источники:
- FineWeb2-HQ / rus_Cyrl: потоковый отбор документов, без скачивания всего архива.
- OpenAssistant/oasst1: пары промпт/ответ с русским языком и прошедшими review.

Скрипт не загружает чужие веса и не вызывает внешние AI API.
"""
import argparse
import html
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")
REJECT_LABELS = {
    "spam",
    "lang_mismatch",
    "pii",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
    "fails_task",
}


def clean_text(value: str) -> str:
    """Normalize text and mask common direct identifiers."""
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = text.replace("\x00", " ")
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = IPV4_RE.sub("[IP]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def russian_letter_ratio(text: str) -> float:
    letters = LETTER_RE.findall(text)
    if not letters:
        return 0.0
    return len(CYRILLIC_RE.findall(text)) / len(letters)


def labels_have_rejection(labels) -> bool:
    """Reject OASST rows explicitly flagged by annotators.

    Hugging Face currently exposes OASST1 labels as parallel arrays:
    {"name": ["spam", "pii"], "value": [0.0, 0.25], "count": [3, 3]}.
    Also accept transformed exports mapping names to value objects.
    """
    if not isinstance(labels, dict):
        return False

    if "name" in labels and "value" in labels:
        names = labels.get("name", []) or []
        values = labels.get("value", []) or []
        candidates = zip(names, values)
    else:
        candidates = labels.items()

    for name, details in candidates:
        if str(name).lower() not in REJECT_LABELS:
            continue
        value = details.get("value", 0) if isinstance(details, dict) else details
        try:
            if float(value) > 0.0:
                return True
        except (TypeError, ValueError):
            if value:
                return True
    return False


def valid_oasst_row(row: dict) -> bool:
    if row.get("lang") != "ru":
        return False
    if row.get("deleted") is True or row.get("synthetic") is True:
        return False
    if row.get("review_result") is not True:
        return False
    if labels_have_rejection(row.get("labels")):
        return False
    text = clean_text(row.get("text", ""))
    return 12 <= len(text) <= 2400


def extract_oasst_pairs(rows: Iterable[dict], max_pairs: int, seed: int = 17):
    """Pair Russian assistant replies with their reviewed Russian parent prompts."""
    by_id = {
        row.get("message_id"): row
        for row in rows
        if row.get("message_id")
    }
    pairs = []
    seen = set()
    for answer in by_id.values():
        if answer.get("role") != "assistant" or not valid_oasst_row(answer):
            continue
        prompt = by_id.get(answer.get("parent_id"))
        if prompt is None or prompt.get("role") != "prompter" or not valid_oasst_row(prompt):
            continue
        question_text = clean_text(prompt.get("text", ""))
        answer_text = clean_text(answer.get("text", ""))
        if russian_letter_ratio(question_text) < 0.25 or russian_letter_ratio(answer_text) < 0.25:
            continue
        key = (question_text.casefold(), answer_text.casefold())
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"question": question_text, "answer": answer_text})
    random.Random(seed).shuffle(pairs)
    return pairs[:max_pairs]


def split_items(items: list, validation_fraction: float, seed: int):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def load_fineweb_documents(max_documents: int, max_chars: int, max_doc_chars: int,
                           min_doc_chars: int, validation_fraction: float, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Для загрузки датасетов установи пакет datasets.") from error

    stream = load_dataset(
        "epfml/FineWeb2-HQ",
        "rus_Cyrl",
        split="train",
        streaming=True,
    )
    stream = stream.shuffle(seed=seed, buffer_size=1000)
    train_docs, validation_docs = [], []
    total_chars = 0
    rng = random.Random(seed)

    for row in stream:
        text = clean_text(row.get("text", ""))
        if len(text) < min_doc_chars:
            continue
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars].rsplit(" ", 1)[0]
        if russian_letter_ratio(text) < 0.35:
            continue
        language_score = row.get("language_score")
        if language_score is not None and float(language_score) < 0.85:
            continue
        # FineWeb2-HQ уже является отбором top-10% по оценке качества.
        # Если документ помечен как часть дубликатного кластера, пропускаем его.
        cluster_size = row.get("minhash_cluster_size")
        if cluster_size is not None and int(cluster_size) > 1:
            continue
        if total_chars + len(text) > max_chars:
            break
        total_chars += len(text)
        target = validation_docs if rng.random() < validation_fraction else train_docs
        target.append(text)
        if len(train_docs) + len(validation_docs) >= max_documents:
            break

    if not train_docs:
        raise RuntimeError("FineWeb2-HQ не дал пригодных русскоязычных документов.")
    return train_docs, validation_docs, total_chars


def write_text(path: Path, documents: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n<|document|>\n\n".join(documents) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare capped Russian training samples from open datasets.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--web-documents", type=int, default=1500)
    parser.add_argument("--web-max-chars", type=int, default=800000)
    parser.add_argument("--max-doc-chars", type=int, default=3000)
    parser.add_argument("--min-doc-chars", type=int, default=250)
    parser.add_argument("--oasst-max-pairs", type=int, default=1800)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.web_documents < 10 or args.web_max_chars < 10000:
        parser.error("Лимит веб-корпуса слишком мал.")
    if not 0.05 <= args.validation_fraction <= 0.2:
        parser.error("--validation-fraction должен быть от 0.05 до 0.2.")

    output = Path(args.output_dir)
    web_train, web_valid, web_chars = load_fineweb_documents(
        args.web_documents,
        args.web_max_chars,
        args.max_doc_chars,
        args.min_doc_chars,
        args.validation_fraction,
        args.seed,
    )

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Для загрузки OASST1 установи пакет datasets.") from error

    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    pairs = extract_oasst_pairs(oasst, args.oasst_max_pairs, args.seed)
    if len(pairs) < 2:
        raise RuntimeError("Не удалось получить достаточно проверенных русских пар из OASST1.")
    sft_train, sft_valid = split_items(pairs, args.validation_fraction, args.seed)

    write_text(output / "fineweb2_ru_train.txt", web_train)
    write_text(output / "fineweb2_ru_validation.txt", web_valid)
    write_jsonl(output / "openassistant_ru_train.jsonl", sft_train)
    write_jsonl(output / "openassistant_ru_validation.jsonl", sft_valid)

    manifest = {
        "seed": args.seed,
        "fineweb2_hq": {
            "dataset": "epfml/FineWeb2-HQ",
            "config": "rus_Cyrl",
            "license": "ODC-By-1.0; subject to CommonCrawl terms",
            "train_documents": len(web_train),
            "validation_documents": len(web_valid),
            "total_characters_capped": web_chars,
        },
        "openassistant": {
            "dataset": "OpenAssistant/oasst1",
            "license": "Apache-2.0",
            "train_pairs": len(sft_train),
            "validation_pairs": len(sft_valid),
            "filters": [
                "lang=ru",
                "review_result=true",
                "not deleted or synthetic",
                "paired assistant message with its parent prompter message",
                "drop rows with positive spam/PII/hate/sexual-content/etc. review labels",
                "mask obvious emails, IPv4 addresses and phone-like strings",
            ],
        },
        "notice": (
            "Streaming sample with strict caps, not the full FineWeb2-HQ corpus. "
            "Regex masking cannot guarantee removal of every personal identifier."
        ),
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
