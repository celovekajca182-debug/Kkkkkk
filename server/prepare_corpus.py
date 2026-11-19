"""Сбор небольшого проверяемого русскоязычного корпуса из открытых датасетов.

Источники:
- FineWeb2-HQ / rus_Cyrl: потоковый отбор документов, без скачивания всего архива.
- OpenAssistant/oasst1: пары промпт/ответ с русским языком и прошедшими review.

Скрипт не загружает чужие веса и не вызывает внешние AI API.
"""
import argparse
import hashlib
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

OASST_LANGUAGE_CODES = ("ru", "en", "de", "es", "fr", "zh", "uk", "pt", "it", "ja")
FINEWEB2_MULTILINGUAL_CONFIGS = {
    "deu_Latn": "de",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "cmn_Hani": "zh",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "jpn_Jpan": "ja",
}
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


def script_letter_ratio(text: str, language: str) -> float:
    """Measure the share of letters belonging to the script expected for a language."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    language = language.lower().split("-", 1)[0]
    if language in {"ru", "uk"}:
        return sum("CYRILLIC" in unicodedata.name(char, "") for char in letters) / len(letters)
    if language == "zh":
        return sum(
            "CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "")
            or "CJK COMPATIBILITY IDEOGRAPH" in unicodedata.name(char, "")
            for char in letters
        ) / len(letters)
    if language == "ja":
        return sum(
            "HIRAGANA" in unicodedata.name(char, "")
            or "KATAKANA" in unicodedata.name(char, "")
            or "CJK UNIFIED IDEOGRAPH" in unicodedata.name(char, "")
            or "CJK COMPATIBILITY IDEOGRAPH" in unicodedata.name(char, "")
            for char in letters
        ) / len(letters)
    return sum("LATIN" in unicodedata.name(char, "") for char in letters) / len(letters)


def normalize_oasst_language(value: object) -> str:
    candidate = str(value or "").strip().lower().replace("_", "-")
    candidate = candidate.split("-", 1)[0]
    return candidate if candidate in OASST_LANGUAGE_CODES else ""


def truncate_document(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Word-boundary truncation is useful for spaced scripts; do not empty CJK text
    # merely because a document has no ASCII spaces.
    boundary = truncated.rfind(" ")
    if boundary >= max_chars // 2:
        truncated = truncated[:boundary]
    return truncated.strip()


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
    if normalize_oasst_language(row.get("lang")) not in OASST_LANGUAGE_CODES:
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
    """Pair reviewed OASST prompts/replies in ten represented languages.

    Samples are balanced round-robin so English or Spanish cannot crowd out
    Russian, German, French, Chinese, Ukrainian, Portuguese, Italian or Japanese.
    """
    by_id = {
        row.get("message_id"): row
        for row in rows
        if row.get("message_id")
    }
    buckets: dict[str, list[dict]] = {language: [] for language in OASST_LANGUAGE_CODES}
    seen = set()
    for answer in by_id.values():
        if answer.get("role") != "assistant" or not valid_oasst_row(answer):
            continue
        prompt = by_id.get(answer.get("parent_id"))
        if prompt is None or prompt.get("role") != "prompter" or not valid_oasst_row(prompt):
            continue
        language = normalize_oasst_language(prompt.get("lang"))
        if language != normalize_oasst_language(answer.get("lang")):
            continue
        question_text = clean_text(prompt.get("text", ""))
        answer_text = clean_text(answer.get("text", ""))
        if (
            script_letter_ratio(question_text, language) < 0.18
            or script_letter_ratio(answer_text, language) < 0.18
        ):
            continue
        key = (language, question_text.casefold(), answer_text.casefold())
        if key in seen:
            continue
        seen.add(key)
        buckets[language].append({
            "question": question_text,
            "answer": answer_text,
            "language": language,
        })

    for index, language in enumerate(OASST_LANGUAGE_CODES):
        random.Random(seed + index).shuffle(buckets[language])

    selected: list[dict] = []
    positions = {language: 0 for language in OASST_LANGUAGE_CODES}
    while len(selected) < max_pairs:
        added_this_round = False
        for language in OASST_LANGUAGE_CODES:
            position = positions[language]
            if position < len(buckets[language]) and len(selected) < max_pairs:
                selected.append(buckets[language][position])
                positions[language] += 1
                added_this_round = True
        if not added_this_round:
            break
    return selected


def split_items(items: list, validation_fraction: float, seed: int):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        return shuffled, []
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def load_fineweb_language_documents(
    config: str,
    language: str,
    max_documents: int,
    max_chars: int,
    max_doc_chars: int,
    min_doc_chars: int,
    validation_fraction: float,
    seed: int,
):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Для загрузки датасетов установи пакет datasets.") from error

    stream = load_dataset(
        "epfml/FineWeb2-HQ",
        config,
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=1000)
    train_docs: list[str] = []
    validation_docs: list[str] = []
    total_chars = 0
    rng = random.Random(seed)

    for row in stream:
        text = clean_text(row.get("text", ""))
        if len(text) < min_doc_chars:
            continue
        text = truncate_document(text, max_doc_chars)
        if script_letter_ratio(text, language) < 0.35:
            continue
        language_score = row.get("language_score")
        if language_score is not None and float(language_score) < 0.85:
            continue
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
        raise RuntimeError(f"FineWeb2-HQ не дал пригодных документов для языка {language}.")
    return train_docs, validation_docs, total_chars


def load_fineweb_documents(
    max_documents: int,
    max_chars: int,
    max_doc_chars: int,
    min_doc_chars: int,
    validation_fraction: float,
    seed: int,
):
    return load_fineweb_language_documents(
        "rus_Cyrl", "ru", max_documents, max_chars, max_doc_chars,
        min_doc_chars, validation_fraction, seed,
    )


def load_finewiki_documents(
    max_documents: int,
    max_chars: int,
    max_doc_chars: int,
    min_doc_chars: int,
    validation_fraction: float,
    seed: int,
):
    """Stream a reproducible Russian Wikipedia sample from FineWiki."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Для загрузки FineWiki установи пакет datasets.") from error

    stream = load_dataset(
        "HuggingFaceFW/finewiki",
        "ru",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=500)

    train_docs: list[str] = []
    validation_docs: list[str] = []
    seen_titles: set[str] = set()
    seen_texts: set[str] = set()
    total_chars = 0
    rng = random.Random(seed)

    for row in stream:
        title = clean_text(row.get("title", ""))
        body = clean_text(row.get("text", ""))
        if not title or len(body) < min_doc_chars:
            continue
        if russian_letter_ratio(title + " " + body) < 0.35:
            continue

        # FineWiki already removes most citation sections. Drop any remaining
        # trailing reference-only headings without destroying article structure.
        body = re.split(
            r"(?im)^\s{0,3}(?:#{1,6}\s*)?(?:Примечания|Литература|Ссылки|Источники|См\. также)\s*$",
            body,
            maxsplit=1,
        )[0].strip()
        if len(body) < min_doc_chars:
            continue

        title_key = title.casefold()
        if title_key in seen_titles:
            continue
        if len(body) > max_doc_chars:
            body = body[:max_doc_chars].rsplit(" ", 1)[0].strip()
        document = title + "\n" + body
        fingerprint = hashlib.sha256(document.encode("utf-8")).hexdigest()
        if fingerprint in seen_texts:
            continue
        if total_chars + len(document) > max_chars:
            break

        seen_titles.add(title_key)
        seen_texts.add(fingerprint)
        total_chars += len(document)
        target = validation_docs if rng.random() < validation_fraction else train_docs
        target.append(document)
        if len(train_docs) + len(validation_docs) >= max_documents:
            break

    if not train_docs or not validation_docs:
        raise RuntimeError(
            "FineWiki не дал достаточного числа русских статей для train и validation."
        )
    return train_docs, validation_docs, total_chars


def load_classic_documents(
    max_documents: int,
    max_chars: int,
    max_doc_chars: int,
    min_doc_chars: int,
    validation_fraction: float,
    seed: int,
):
    """Stream a small public-domain Russian classics sample for language pretraining."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Для загрузки ru-classic установи пакет datasets.") from error

    stream = load_dataset(
        "Imperius/ru-classic",
        split="train",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=500)

    train_docs: list[str] = []
    validation_docs: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    rng = random.Random(seed)

    for row in stream:
        text = clean_text(row.get("text") or row.get("content") or "")
        if len(text) < min_doc_chars or russian_letter_ratio(text) < 0.4:
            continue
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars].rsplit(" ", 1)[0].strip()
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if fingerprint in seen:
            continue
        if total_chars + len(text) > max_chars:
            break
        seen.add(fingerprint)
        total_chars += len(text)
        target = validation_docs if rng.random() < validation_fraction else train_docs
        target.append(text)
        if len(train_docs) + len(validation_docs) >= max_documents:
            break

    if not train_docs or not validation_docs:
        raise RuntimeError("ru-classic не дал достаточного числа русских текстов для train/validation.")
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
    parser.add_argument("--wiki-documents", type=int, default=1200)
    parser.add_argument("--wiki-max-chars", type=int, default=600000)
    parser.add_argument("--classic-documents", type=int, default=400)
    parser.add_argument("--classic-max-chars", type=int, default=250000)
    parser.add_argument(
        "--multilingual-web-languages",
        default="deu_Latn,spa_Latn,fra_Latn,cmn_Hani,ita_Latn,por_Latn,jpn_Jpan",
    )
    parser.add_argument("--multilingual-web-documents", type=int, default=300)
    parser.add_argument("--multilingual-web-max-chars", type=int, default=100000)
    parser.add_argument("--max-doc-chars", type=int, default=3000)
    parser.add_argument("--min-doc-chars", type=int, default=250)
    parser.add_argument("--oasst-max-pairs", type=int, default=1800)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.web_documents < 10 or args.web_max_chars < 10000:
        parser.error("Лимит веб-корпуса слишком мал.")
    if args.wiki_documents < 10 or args.wiki_max_chars < 10000:
        parser.error("Лимит Wikipedia-корпуса слишком мал.")
    if args.classic_documents < 10 or args.classic_max_chars < 10000:
        parser.error("Лимит ru-classic-корпуса слишком мал.")
    if args.multilingual_web_documents < 10 or args.multilingual_web_max_chars < 10000:
        parser.error("Лимит дополнительного мультиязычного веб-корпуса слишком мал.")
    requested_configs = [item.strip() for item in args.multilingual_web_languages.split(",") if item.strip()]
    unknown_configs = sorted(set(requested_configs) - set(FINEWEB2_MULTILINGUAL_CONFIGS))
    if unknown_configs:
        parser.error("Неподдерживаемые FineWeb2-HQ конфиги: " + ", ".join(unknown_configs))
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
    wiki_train, wiki_valid, wiki_chars = load_finewiki_documents(
        args.wiki_documents,
        args.wiki_max_chars,
        args.max_doc_chars,
        args.min_doc_chars,
        args.validation_fraction,
        args.seed + 1,
    )
    classic_train, classic_valid, classic_chars = load_classic_documents(
        args.classic_documents,
        args.classic_max_chars,
        args.max_doc_chars,
        args.min_doc_chars,
        args.validation_fraction,
        args.seed + 2,
    )
    multilingual_web = {}
    for index, config in enumerate(requested_configs):
        language = FINEWEB2_MULTILINGUAL_CONFIGS[config]
        train_docs, validation_docs, char_count = load_fineweb_language_documents(
            config,
            language,
            args.multilingual_web_documents,
            args.multilingual_web_max_chars,
            args.max_doc_chars,
            args.min_doc_chars,
            args.validation_fraction,
            args.seed + 10 + index,
        )
        multilingual_web[language] = {
            "config": config,
            "train": train_docs,
            "validation": validation_docs,
            "characters": char_count,
        }

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
    write_text(output / "finewiki_ru_train.txt", wiki_train)
    write_text(output / "finewiki_ru_validation.txt", wiki_valid)
    write_text(output / "ru_classic_train.txt", classic_train)
    write_text(output / "ru_classic_validation.txt", classic_valid)
    for language, documents in multilingual_web.items():
        language_dir = output / "multilingual"
        write_text(language_dir / f"fineweb2_{language}_train.txt", documents["train"])
        write_text(language_dir / f"fineweb2_{language}_validation.txt", documents["validation"])
    write_jsonl(output / "openassistant_multilingual_train.jsonl", sft_train)
    write_jsonl(output / "openassistant_multilingual_validation.jsonl", sft_valid)

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
        "finewiki": {
            "dataset": "HuggingFaceFW/finewiki",
            "config": "ru",
            "source": "https://huggingface.co/datasets/HuggingFaceFW/finewiki",
            "upstream": "https://ru.wikipedia.org/",
            "license": "CC-BY-SA-4.0 (processed adaptation; GFDL also applies to relevant source contributions)",
            "attribution": "Wikipedia contributors; FineWiki dataset by HuggingFaceFW/FineData",
            "train_documents": len(wiki_train),
            "validation_documents": len(wiki_valid),
            "total_characters_capped": wiki_chars,
            "filters": [
                "Russian Cyrillic share >= 35%",
                "minimum article length",
                "deduplicate article titles and content",
                "remove trailing reference-only sections when present",
                "streaming sample with deterministic seed",
            ],
        },
        "ru_classic": {
            "dataset": "Imperius/ru-classic",
            "source": "https://huggingface.co/datasets/Imperius/ru-classic",
            "license": "CC0-1.0 compilation; source texts are public domain according to dataset card",
            "train_documents": len(classic_train),
            "validation_documents": len(classic_valid),
            "total_characters_capped": classic_chars,
            "purpose": "small style/language supplement only; not a modern factual QA source",
        },
        "fineweb2_multilingual": {
            "dataset": "epfml/FineWeb2-HQ",
            "license": "ODC-By-1.0; subject to CommonCrawl terms",
            "languages": {
                language: {
                    "config": documents["config"],
                    "train_documents": len(documents["train"]),
                    "validation_documents": len(documents["validation"]),
                    "total_characters_capped": documents["characters"],
                }
                for language, documents in multilingual_web.items()
            },
        },
        "openassistant": {
            "dataset": "OpenAssistant/oasst1",
            "license": "Apache-2.0",
            "train_pairs": len(sft_train),
            "validation_pairs": len(sft_valid),
            "train_pairs_by_language": {
                language: sum(pair.get("language") == language for pair in sft_train)
                for language in OASST_LANGUAGE_CODES
            },
            "filters": [
                "lang in ru/en/de/es/fr/zh/uk/pt/it/ja",
                "review_result=true",
                "not deleted or synthetic",
                "paired assistant message with its parent prompter message",
                "drop rows with positive spam/PII/hate/sexual-content/etc. review labels",
                "mask obvious emails, IPv4 addresses and phone-like strings",
            ],
        },
        "notice": (
            "Streaming samples with strict caps, not the full source datasets. "
            "Regex masking cannot guarantee removal of every personal identifier. "
            "Wikipedia/FineWiki text is CC BY-SA 4.0 and requires attribution; "
            "this prepared adaptation must retain the source license and notices."
        ),
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
