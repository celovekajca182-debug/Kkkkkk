"""Prepare an attributed Russian FineWiki corpus for SWIN's runtime retrieval index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_corpus import load_finewiki_documents, write_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a capped Russian FineWiki sample for SWIN retrieval."
    )
    parser.add_argument("--documents", type=int, default=1500)
    parser.add_argument("--max-chars", type=int, default=900000)
    parser.add_argument("--max-doc-chars", type=int, default=2500)
    parser.add_argument("--min-doc-chars", type=int, default=250)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="server/data/finewiki_ru_train.txt")
    args = parser.parse_args()

    if args.documents < 40 or args.max_chars < 30000:
        parser.error("The FineWiki retrieval sample is too small.")
    train_docs, validation_docs, total_chars = load_finewiki_documents(
        args.documents,
        args.max_chars,
        args.max_doc_chars,
        args.min_doc_chars,
        args.validation_fraction,
        args.seed,
    )
    output = Path(args.output)
    write_text(output, train_docs)
    size = output.stat().st_size
    if size < 1000 or size > 32 * 1024 * 1024:
        raise RuntimeError(f"Prepared FineWiki corpus has an unexpected size: {size} bytes.")

    report = {
        "dataset": "HuggingFaceFW/finewiki",
        "config": "ru",
        "upstream": "https://ru.wikipedia.org/",
        "train_articles": len(train_docs),
        "validation_articles_not_published": len(validation_docs),
        "characters": total_chars,
        "bytes": size,
        "license": "CC-BY-SA-4.0; GFDL applies to relevant upstream contributions",
        "attribution": "Wikipedia contributors; FineWiki dataset by HuggingFaceFW/FineData",
        "output": str(output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
