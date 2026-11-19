import argparse
import json
import math
import random
from pathlib import Path

import torch

from transformer import BPEByteTokenizer, CharTokenizer, SWINTransformer, load_transformer, save_transformer


def read_instruction_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        return []
    records: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Ошибка JSONL в {path}, строка {line_number}: {error}") from error
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            records.append((question, answer))
    return records


def read_instruction_file(path: Path) -> list[str]:
    return [
        f"Пользователь: {question}\nSWIN: {answer}"
        for question, answer in read_instruction_pairs(path)
    ]


def build_instruction_examples(
    knowledge_path: Path,
    dialogue_path: Path,
    instructions_path: Path,
    coding_path: Path | None = None,
    project_examples_path: Path | None = None,
) -> list[tuple[str, str]]:
    """Return prompt/answer pairs so SFT can mask the user prompt loss."""
    examples: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(prompt: str, answer: str):
        prompt = prompt.strip()
        answer = answer.strip()
        question_key = prompt.rsplit("Пользователь:", 1)[-1].strip()
        key = (question_key.casefold(), answer.casefold())
        if prompt and answer and key not in seen:
            seen.add(key)
            examples.append((prompt, answer))

    if knowledge_path.is_file():
        entries = json.loads(knowledge_path.read_text(encoding="utf-8"))
        for item in entries:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            title = str(item.get("title", "")).strip()
            if question and answer:
                prefix = f"Тема: {title}\n" if title else ""
                add(prefix + f"Пользователь: {question}", answer)

    for path in (dialogue_path, instructions_path, coding_path, project_examples_path):
        if path is None:
            continue
        for question, answer in read_instruction_pairs(path):
            add(f"Пользователь: {question}", answer)

    return examples


def join_corpus(*pieces: str) -> str:
    return "\n\n".join(piece.strip() for piece in pieces if piece and piece.strip()) + "\n"


def build_pretraining_corpus(
    data_path: Path,
    web_path: Path,
    wiki_path: Path | None = None,
    classic_path: Path | None = None,
) -> str:
    pieces = [
        "Русскоязычный текстовый корпус SWIN AI. Задача модели — предсказывать следующий токен по предыдущему контексту.",
        "Источник: отобранные открытые русские веб-документы.",
    ]
    if data_path.is_file():
        pieces.append(data_path.read_text(encoding="utf-8"))
    if web_path.is_file():
        pieces.append(web_path.read_text(encoding="utf-8"))
    if wiki_path is not None and wiki_path.is_file():
        pieces.extend([
            "Энциклопедические материалы русской Wikipedia (FineWiki).",
            wiki_path.read_text(encoding="utf-8"),
        ])
    if classic_path is not None and classic_path.is_file():
        pieces.extend([
            "Небольшая языковая выборка русской классической литературы (CC0).",
            classic_path.read_text(encoding="utf-8"),
        ])
    return join_corpus(*pieces)


def build_instruction_corpus(
    knowledge_path: Path,
    dialogue_path: Path,
    instructions_path: Path,
    coding_path: Path | None = None,
    project_examples_path: Path | None = None,
) -> str:
    pieces = [
        "Ты — SWIN AI, самостоятельный русскоязычный помощник по программированию, разработке игр и техническим вопросам.",
        "Отвечай прямо, спокойно и по существу. Не выдумывай факты. Если данных недостаточно, честно обозначь это и предложи проверяемый следующий шаг.",
    ]
    for prompt, answer in build_instruction_examples(
        knowledge_path,
        dialogue_path,
        instructions_path,
        coding_path,
        project_examples_path,
    ):
        pieces.append(f"{prompt}\nSWIN: {answer}")
    return join_corpus(*pieces)


def build_sft_windows(
    examples: list[tuple[str, str]],
    tokenizer: CharTokenizer,
    block_size: int,
    prompt_budget: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build fixed-size response-only SFT windows with user prompt loss masked.

    Target positions corresponding to the prompt are set to -100 (PyTorch's
    cross-entropy ignore index). Long prompts are compacted and long answers are
    covered by overlapping windows so every answer section receives supervision.
    """
    if block_size < 32:
        raise ValueError("Для SFT block_size должен быть не меньше 32.")
    prompt_budget = prompt_budget or max(64, block_size // 2)
    prompt_budget = min(prompt_budget, block_size - 8)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = tokenizer.stoi.get(" ", 0)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    answer_suffix = tokenizer.encode("\n\n")
    if eos_token_id is not None:
        answer_suffix.append(eos_token_id)
    windows: list[tuple[torch.Tensor, torch.Tensor]] = []
    stride = max(16, block_size // 2)

    for prompt, answer in examples:
        prompt = prompt.strip() + "\nSWIN:"
        answer = answer.strip()
        if not prompt or not answer:
            continue

        prompt_ids = tokenizer.encode(prompt)
        if len(prompt_ids) > prompt_budget:
            # Preserve the role/task opening and the most recent question context.
            head_size = min(24, prompt_budget // 4)
            tail_size = prompt_budget - head_size
            prompt_ids = prompt_ids[:head_size] + prompt_ids[-tail_size:]

        answer_ids = tokenizer.encode(answer)
        if not answer_ids:
            continue
        sequence = prompt_ids + answer_ids + answer_suffix
        prompt_end = len(prompt_ids)
        answer_end = prompt_end + len(answer_ids)

        start = 0
        while start < answer_end:
            chunk = sequence[start:start + block_size + 1]
            if len(chunk) < 2:
                break

            input_ids = chunk[:-1]
            target_ids = chunk[1:]
            targets = []
            for offset, token_id in enumerate(target_ids):
                absolute_target_position = start + offset + 1
                if absolute_target_position < prompt_end:
                    targets.append(-100)
                else:
                    targets.append(token_id)

            if any(token_id != -100 for token_id in targets):
                input_padded = input_ids + [pad_id] * (block_size - len(input_ids))
                target_padded = targets + [-100] * (block_size - len(targets))
                windows.append((
                    torch.tensor(input_padded, dtype=torch.long),
                    torch.tensor(target_padded, dtype=torch.long),
                ))

            if start + block_size + 1 >= len(sequence):
                break
            start += stride

    if not windows:
        raise ValueError("Не удалось построить SFT-окна с токенами ответа.")
    return windows


def sample_sft_batch(
    windows: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    device: torch.device,
):
    if not windows:
        raise ValueError("SFT dataset не содержит обучающих окон.")
    choices = torch.randint(0, len(windows), (batch_size,)).tolist()
    x = torch.stack([windows[index][0] for index in choices])
    y = torch.stack([windows[index][1] for index in choices])
    if not torch.any(y != -100):
        raise ValueError("SFT batch не содержит ни одного target токена.")
    return x.to(device), y.to(device)


def build_tensor_splits(
    train_corpus: str,
    validation_corpus: str,
    block_size: int,
    tokenizer=None,
):
    if tokenizer is None:
        tokenizer = CharTokenizer.from_texts([train_corpus, validation_corpus])

    train_encoded = tokenizer.encode(train_corpus)
    if len(train_encoded) < block_size * 4:
        raise ValueError("Корпус этапа обучения слишком мал. Добавь больше проверенных данных.")
    if len(set(train_encoded)) < 20:
        raise ValueError("В корпусе слишком мало разных символов.")

    if validation_corpus.strip():
        validation_encoded = tokenizer.encode(validation_corpus)
        if len(validation_encoded) <= block_size + 1:
            raise ValueError("Отложенный validation-корпус слишком мал.")
        train_data = torch.tensor(train_encoded, dtype=torch.long)
        val_data = torch.tensor(validation_encoded, dtype=torch.long)
    else:
        split = max(block_size + 2, int(len(train_encoded) * 0.9))
        split = min(split, len(train_encoded) - block_size - 2)
        train_data = torch.tensor(train_encoded[:split], dtype=torch.long)
        val_data = torch.tensor(train_encoded[split:], dtype=torch.long)

    return train_data, val_data


def sample_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    if data.numel() <= block_size + 1:
        raise ValueError("Недостаточно данных для формирования обучающего блока.")
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[start:start + block_size] for start in starts])
    y = torch.stack([data[start + 1:start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    batch_size,
    block_size,
    device,
    steps=8,
    train_windows=None,
    val_windows=None,
):
    model.eval()
    result = {}
    for name, data, windows in (
        ("train", train_data, train_windows),
        ("validation", val_data, val_windows),
    ):
        losses = []
        for _ in range(steps):
            if windows is not None:
                x, y = sample_sft_batch(windows, batch_size, device)
            else:
                x, y = sample_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def train_stage(
    model,
    train_data,
    val_data,
    epochs: int,
    batch_size: int,
    block_size: int,
    device,
    output_path: Path,
    stage_name: str,
    train_windows=None,
    val_windows=None,
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    training_size = len(train_windows) if train_windows is not None else len(train_data)
    units_per_batch = batch_size if train_windows is not None else batch_size * block_size
    batches_per_epoch = max(
        10,
        min(80, math.ceil(training_size / units_per_batch) * 2),
    )
    total_steps = max(1, epochs * batches_per_epoch)
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(0.1, (step + 1) / warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    best_validation = float("inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for _ in range(batches_per_epoch):
            if train_windows is not None:
                x, y = sample_sft_batch(train_windows, batch_size, device)
            else:
                x, y = sample_batch(train_data, batch_size, block_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        metrics = estimate_loss(
            model,
            train_data,
            val_data,
            batch_size,
            block_size,
            device,
            train_windows=train_windows,
            val_windows=val_windows,
        )
        print(
            f"{stage_name} epoch {epoch}/{epochs} "
            f"train_loss={total_loss / batches_per_epoch:.4f} "
            f"validation_loss={metrics['validation']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if metrics["validation"] < best_validation:
            best_validation = metrics["validation"]
            save_transformer(model.to("cpu"), str(output_path), best_validation)
            model.to(device)

    print(f"{stage_name}: best validation loss={best_validation:.4f}")
    return load_transformer(str(output_path)).to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage training of SWIN AI from random initialization."
    )
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--knowledge", default="data/knowledge.json")
    parser.add_argument("--dialogue", default="data/dialogue.jsonl")
    parser.add_argument("--web-train", default="data/fineweb2_ru_train.txt")
    parser.add_argument("--web-validation", default="data/fineweb2_ru_validation.txt")
    parser.add_argument("--wiki-train", default="data/finewiki_ru_train.txt")
    parser.add_argument("--wiki-validation", default="data/finewiki_ru_validation.txt")
    parser.add_argument("--classic-train", default="data/ru_classic_train.txt")
    parser.add_argument("--classic-validation", default="data/ru_classic_validation.txt")
    parser.add_argument("--instructions-train", default="data/openassistant_ru_train.jsonl")
    parser.add_argument("--instructions-validation", default="data/openassistant_ru_validation.jsonl")
    parser.add_argument("--coding-train", default="data/mbpp_python_train.jsonl")
    parser.add_argument("--coding-validation", default="data/mbpp_python_validation.jsonl")
    parser.add_argument("--project-examples", default="data/coding_dialogue.jsonl")
    parser.add_argument("--output", default="models/swin-transformer.pt")
    parser.add_argument("--pretrain-output", default="models/swin-transformer-pretrain.pt")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--sft-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not 1 <= args.pretrain_epochs <= 200 or not 1 <= args.sft_epochs <= 100:
        parser.error("Недопустимое число эпох.")
    if not 4 <= args.batch_size <= 128:
        parser.error("--batch-size должен быть от 4 до 128")
    if args.d_model % args.heads != 0 or args.block_size < 32:
        parser.error("Проверь --d-model, --heads и --block-size.")
    if not 512 <= args.vocab_size <= 32768:
        parser.error("--vocab-size должен быть от 512 до 32768.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство обучения: {device}")

    # Подготовить обе стадии заранее, чтобы они использовали один и тот же Unicode-словарь.
    pretrain_train = build_pretraining_corpus(
        Path(args.data),
        Path(args.web_train),
        Path(args.wiki_train),
        Path(args.classic_train),
    )
    validation_parts = []
    for validation_path in (
        Path(args.web_validation),
        Path(args.wiki_validation),
        Path(args.classic_validation),
    ):
        if validation_path.is_file():
            validation_parts.append(validation_path.read_text(encoding="utf-8"))
    pretrain_validation = join_corpus(*validation_parts)
    sft_examples = build_instruction_examples(
        Path(args.knowledge),
        Path(args.dialogue),
        Path(args.instructions_train),
        Path(args.coding_train),
        Path(args.project_examples),
    )
    sft_validation_examples = (
        read_instruction_pairs(Path(args.instructions_validation))
        + read_instruction_pairs(Path(args.coding_validation))
    )
    sft_train = build_instruction_corpus(
        Path(args.knowledge),
        Path(args.dialogue),
        Path(args.instructions_train),
        Path(args.coding_train),
        Path(args.project_examples),
    )
    # Never build the vocabulary from held-out validation examples.
    tokenizer = BPEByteTokenizer.from_texts(
        [pretrain_train, sft_train],
        vocab_size=args.vocab_size,
    )
    print(f"Byte-level BPE vocabulary: {tokenizer.vocab_size:,} tokens")

    # Этап 1: языковое предобучение на очищенных русских веб-документах.
    pretrain_train_data, pretrain_val_data = build_tensor_splits(
        pretrain_train, pretrain_validation, args.block_size, tokenizer
    )
    print(
        f"Этап 1 / pretraining: train={pretrain_train_data.numel():,} символов, "
        f"validation={pretrain_val_data.numel():,} символов"
    )

    model = SWINTransformer(
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=0.1,
        vocab_size=tokenizer.vocab_size,
    )
    model.tokenizer = tokenizer
    model = train_stage(
        model,
        pretrain_train_data,
        pretrain_val_data,
        args.pretrain_epochs,
        args.batch_size,
        args.block_size,
        device,
        Path(args.pretrain_output),
        "PRETRAIN",
    )

    # Этап 2: instruction tuning на парах вопрос/ответ с тем же символьным словарём.
    sft_train_windows = build_sft_windows(sft_examples, tokenizer, args.block_size)
    sft_validation_windows = build_sft_windows(
        sft_validation_examples,
        tokenizer,
        args.block_size,
    )
    print(
        f"Этап 2 / instruction tuning: examples={len(sft_examples):,}, "
        f"train_windows={len(sft_train_windows):,}, "
        f"validation_windows={len(sft_validation_windows):,}; "
        "user-prompt loss masked"
    )
    model = train_stage(
        model,
        None,
        None,
        args.sft_epochs,
        args.batch_size,
        args.block_size,
        device,
        Path(args.output),
        "INSTRUCTION",
        train_windows=sft_train_windows,
        val_windows=sft_validation_windows,
    )

    model = model.to("cpu").eval()
    sample = model.generate(
        "Пользователь: Как исправить ошибку Godot?\nSWIN:",
        max_new_tokens=160,
    )
    print(f"Финальные веса SWIN сохранены: {args.output}")
    print("Пробная генерация:", sample[:400])


if __name__ == "__main__":
    main()
