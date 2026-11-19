#!/usr/bin/env python3
"""QLoRA fine-tuning for SWIN AI on hard mathematics and programming tasks.

The legacy SWINTransformer trainer trains a tiny decoder from random initialization.
This script fine-tunes a pretrained Qwen3 checkpoint and saves a PEFT adapter.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

MATH_DATASET = "project-numina/NuminaMath-CoT"
MATH_BENCHMARK = "EleutherAI/hendrycks_math"


def stable_bucket(source: str, question: str) -> int:
    digest = hashlib.sha256((source + "\n" + question).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100


def clean_text(value: object) -> str:
    text = str(value or "").replace("\x00", "").strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def make_record(question: object, answer: object, category: str, source: str, language: str = "en") -> dict | None:
    q = clean_text(question)
    a = clean_text(answer)
    if len(q) < 25 or len(a) < 50 or len(q) > 12000 or len(a) > 16000:
        return None
    if "\ufffd" in q or "\ufffd" in a:
        return None
    return {"question": q, "answer": a, "category": category, "source": source, "language": language}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{number}: {error}") from error
    return rows


def load_numina(max_examples: int, seed: int) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    print("Streaming advanced mathematics:", MATH_DATASET)
    stream = load_dataset(MATH_DATASET, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=12000)
    caps = {
        "olympiads": 1800, "aops_forum": 1200, "amc_aime": 700, "math": 700,
        "synthetic_amc": 700, "synthetic_math": 700, "orca_math": 700,
        "gsm8k": 250, "cn_k12": 250,
    }
    counts: dict[str, int] = {}
    train_rows, validation_rows = [], []
    seen = set()
    for row in stream:
        source = clean_text(row.get("source", "unknown")) or "unknown"
        if source not in caps or counts.get(source, 0) >= caps[source]:
            continue
        item = make_record(row.get("problem"), row.get("solution"), "math", MATH_DATASET + ":" + source)
        if item is None:
            continue
        key = hashlib.sha256(item["question"].casefold().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        counts[source] = counts.get(source, 0) + 1
        (validation_rows if stable_bucket(MATH_DATASET, item["question"]) < 5 else train_rows).append(item)
        if len(train_rows) + len(validation_rows) >= max_examples:
            break
    print("NuminaMath:", len(train_rows), "train /", len(validation_rows), "validation")
    print("Source mix:", json.dumps(counts, ensure_ascii=False))
    return train_rows, validation_rows


def load_hendrycks(max_per_config: int, max_validation_per_config: int) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    configs = [
        "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
        "number_theory", "prealgebra", "precalculus",
    ]
    train_rows, validation_rows = [], []
    for config in configs:
        try:
            train = load_dataset(MATH_BENCHMARK, config, split="train")
            test = load_dataset(MATH_BENCHMARK, config, split="test")
        except Exception as error:
            print("Skipping MATH config", config, ":", error)
            continue
        count = 0
        for row in train:
            item = make_record(row.get("problem"), row.get("solution"), "math", MATH_BENCHMARK + ":" + config)
            if item:
                train_rows.append(item)
                count += 1
            if count >= max_per_config:
                break
        count = 0
        for row in test:
            item = make_record(row.get("problem"), row.get("solution"), "math", MATH_BENCHMARK + ":" + config + ":test")
            if item:
                validation_rows.append(item)
                count += 1
            if count >= max_validation_per_config:
                break
    print("Hendrycks MATH:", len(train_rows), "train /", len(validation_rows), "validation")
    return train_rows, validation_rows


def load_opencode_reasoning(max_train: int, max_validation: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Stream licensed competitive-programming reasoning examples from NVIDIA."""
    from datasets import load_dataset

    dataset_id = "nvidia/OpenCodeReasoning"
    print("Streaming code-reasoning data:", dataset_id)
    stream = load_dataset(dataset_id, "split_0", split="split_0", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=20000)

    accepted_licenses = {"cc-by-4.0", "mit", "apache-2.0"}
    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    seen: set[str] = set()

    for row in stream:
        license_name = clean_text(row.get("license", "")).lower().replace("_", "-")
        if license_name not in accepted_licenses:
            continue

        source_split = clean_text(row.get("split", "")).lower()
        if source_split not in {"train", "valid", "validation"}:
            # In particular, do not ingest the official test split.
            continue

        question = clean_text(row.get("input", ""))
        reasoning = clean_text(row.get("output", ""))
        solution = clean_text(row.get("solution", ""))
        if not question or not reasoning or not solution:
            continue
        if len(solution) > 12000 or len(reasoning) > 14000:
            continue
        try:
            ast.parse(solution)
        except SyntaxError:
            continue

        # Keep the model explanation plus a clearly separated final code section.
        answer = reasoning
        if solution not in reasoning:
            answer += "\n\nFinal Python implementation:\n" + solution

        item = make_record(
            question,
            answer,
            "coding",
            dataset_id + ":" + clean_text(row.get("source", "unknown")),
        )
        if item is None:
            continue

        key = hashlib.sha256(item["question"].casefold().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)

        use_validation = source_split in {"valid", "validation"} or (
            source_split == "train" and stable_bucket(dataset_id, item["question"]) < 5
        )
        if use_validation:
            if len(validation_rows) < max_validation:
                validation_rows.append(item)
        else:
            if len(train_rows) < max_train:
                train_rows.append(item)

        if len(train_rows) >= max_train and len(validation_rows) >= max_validation:
            break

    print(
        "OpenCodeReasoning selected:",
        len(train_rows), "train /", len(validation_rows), "validation",
    )
    return train_rows, validation_rows


def load_local_code(data_dir: Path) -> tuple[list[dict], list[dict]]:
    train_rows, validation_rows = [], []
    for row in read_jsonl(data_dir / "mbpp_python_train.jsonl"):
        item = make_record(
            row.get("question") or row.get("prompt") or row.get("text"),
            row.get("answer") or row.get("code"),
            "coding", "google-research-datasets/mbpp:sanitized",
        )
        if item:
            train_rows.append(item)
    for row in read_jsonl(data_dir / "mbpp_python_validation.jsonl"):
        item = make_record(
            row.get("question") or row.get("prompt") or row.get("text"),
            row.get("answer") or row.get("code"),
            "coding", "google-research-datasets/mbpp:sanitized:validation",
        )
        if item:
            validation_rows.append(item)

    # Reviewed project samples add GDScript, Kotlin, JavaScript and Android tasks.
    for row in read_jsonl(data_dir / "coding_dialogue.jsonl"):
        item = make_record(
            row.get("question") or row.get("prompt"),
            row.get("answer") or row.get("code"),
            "coding", "SWIN-reviewed-coding-dialogue", str(row.get("language", "ru")),
        )
        if item:
            target = validation_rows if stable_bucket("SWIN-reviewed-coding-dialogue", item["question"]) < 10 else train_rows
            target.append(item)
    print("Programming:", len(train_rows), "train /", len(validation_rows), "validation")
    return train_rows, validation_rows


def deduplicate(rows: Iterable[dict]) -> list[dict]:
    output, seen = [], set()
    for row in rows:
        key = hashlib.sha256(row["question"].casefold().encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def format_prompt(row: dict) -> str:
    if row["category"] == "math":
        return (
            "Solve this challenging mathematics problem. Give a rigorous step-by-step "
            "derivation, justify key transformations, check the result, and clearly state "
            "the final answer.\n\nProblem:\n" + row["question"]
        )
    language_hint = "Пиши объяснения по-русски. " if row.get("language") == "ru" else ""
    return (
        language_hint
        + "Act as a senior software engineer. Produce correct, maintainable code, respect "
        "the requested language and constraints, explain non-obvious decisions, and include "
        "or preserve tests when supplied.\n\nTask:\n" + row["question"]
    )


def format_answer(row: dict) -> str:
    if row["category"] == "math":
        return row["answer"] + "\n\nCheck that every step follows from the conditions and that the final answer is consistent."
    return row["answer"]


class ChatSFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.items = []
        skipped = 0
        for row in rows:
            messages = [
                {"role": "user", "content": format_prompt(row)},
                {"role": "assistant", "content": format_answer(row)},
            ]
            prompt_ids = tokenizer.apply_chat_template(
                messages[:1], tokenize=True, add_generation_prompt=True, enable_thinking=True
            )
            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, enable_thinking=True
            )
            prompt_ids, full_ids = list(prompt_ids), list(full_ids)
            prefix_length = 0
            for left, right in zip(prompt_ids, full_ids):
                if left != right:
                    break
                prefix_length += 1
            if prefix_length < min(8, len(prompt_ids)):
                skipped += 1
                continue
            full_ids = full_ids[:max_length]
            if prefix_length >= len(full_ids) - 8:
                skipped += 1
                continue
            labels = list(full_ids)
            for index in range(min(prefix_length, len(labels))):
                labels[index] = -100
            self.items.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
        if not self.items:
            raise RuntimeError("No usable SFT examples. Check the tokenizer chat template.")
        print("Tokenized:", len(self.items), "examples; skipped:", skipped)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CausalPaddingCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict]) -> dict:
        return {
            "input_ids": pad_sequence([x["input_ids"] for x in features], batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence([x["attention_mask"] for x in features], batch_first=True, padding_value=0),
            "labels": pad_sequence([x["labels"] for x in features], batch_first=True, padding_value=-100),
        }


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for advanced mathematics and programming.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="models/swin-qwen3-1.7b-qlora")
    parser.add_argument("--max-numina", type=int, default=5000)
    parser.add_argument("--max-opencode-train", type=int, default=3000)
    parser.add_argument("--max-opencode-validation", type=int, default=300)
    parser.add_argument("--max-hendrycks-per-config", type=int, default=250)
    parser.add_argument("--max-hendrycks-validation-per-config", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. In Colab choose Runtime > Change runtime type > T4 GPU.")
    if args.max_seq_length < 512:
        parser.error("--max-seq-length must be at least 512.")
    if args.push_to_hub and "/" not in args.hub_model_id:
        parser.error("Set --hub-model-id USERNAME/swin-ai-math-code.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    numina_train, numina_validation = load_numina(args.max_numina, args.seed)
    math_train, math_validation = load_hendrycks(
        args.max_hendrycks_per_config, args.max_hendrycks_validation_per_config
    )
    opencode_train, opencode_validation = load_opencode_reasoning(
        args.max_opencode_train, args.max_opencode_validation, args.seed
    )
    code_train, code_validation = load_local_code(data_dir)

    train_rows = deduplicate(numina_train + math_train + opencode_train + code_train)
    validation_rows = deduplicate(
        numina_validation + math_validation + opencode_validation + code_validation
    )
    coding_train = [row for row in train_rows if row["category"] == "coding"]
    math_train_rows = [row for row in train_rows if row["category"] == "math"]
    if not math_train_rows or not coding_train:
        raise SystemExit(
            "Need both math and coding examples. Run: "
            "python prepare_coding_data.py --max-train 700 --max-validation 160"
        )
    if not validation_rows:
        raise SystemExit("No held-out validation examples; refusing to train without evaluation.")

    # Two passes over coding samples give programming more weight without greeting filler.
    train_rows = math_train_rows + coding_train + coding_train
    print("Final curriculum — math:", len(math_train_rows), "| coding:", len(coding_train) * 2)
    print("  NVIDIA OpenCodeReasoning:", len(opencode_train), "train /", len(opencode_validation), "validation")
    print("Validation:", len(validation_rows), "| generic greeting samples: 0")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = ChatSFTDataset(train_rows, tokenizer, args.max_seq_length)
    eval_dataset = ChatSFTDataset(validation_rows, tokenizer, args.max_seq_length)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading pretrained base:", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    ))
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.04,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id or None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalPaddingCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "base_model": args.base_model,
        "method": "4-bit NF4 QLoRA; LoRA r=16, alpha=32, target_modules=all-linear",
        "train_tokenized_examples": len(train_dataset),
        "validation_tokenized_examples": len(eval_dataset),
        "math_examples_before_tokenization": len(math_train_rows),
        "coding_examples_before_tokenization": len(coding_train),
        "greeting_examples": 0,
        "max_sequence_length": args.max_seq_length,
        "epochs": args.epochs,
        "sources": [
            {"id": MATH_DATASET, "license": "Apache-2.0"},
            {"id": MATH_BENCHMARK, "license": "MIT"},
            {"id": "nvidia/OpenCodeReasoning", "configuration": "split_0", "license": "CC-BY-4.0", "filter": "train/valid only; test excluded"},
            {"id": "google-research-datasets/mbpp", "configuration": "sanitized", "license": "CC-BY-4.0"},
            {"id": "local SWIN coding_dialogue.jsonl", "license": "review required"},
        ],
        "warning": "Experimental adapter. Validation loss does not prove mathematical correctness or successful code execution.",
    }
    (output_dir / "swin_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "---\nbase_model: " + args.base_model + "\nlibrary_name: peft\npipeline_tag: text-generation\nlicense: apache-2.0\n---\n\n"
        "# SWIN AI — mathematics and coding adapter\n\n"
        "Experimental QLoRA adapter for advanced mathematics and programming. "
        "The base model weights are not duplicated in this repository. "
        "See swin_training_manifest.json for data sources and limitations.\n",
        encoding="utf-8",
    )
    print("Adapter saved to:", output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Publish SWIN AI math and coding QLoRA adapter")
        tokenizer.push_to_hub(args.hub_model_id)
        print("Published adapter:", args.hub_model_id)


if __name__ == "__main__":
    main()
