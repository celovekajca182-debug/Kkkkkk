import argparse
import json
import math
import random
from pathlib import Path

import torch

from transformer import ByteTokenizer, SWINTransformer, load_transformer, save_transformer


def read_instruction_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    records = []
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
            records.append(f"Пользователь: {question}\nSWIN: {answer}")
    return records


def join_corpus(*pieces: str) -> str:
    return "\n\n".join(piece.strip() for piece in pieces if piece and piece.strip()) + "\n"


def build_pretraining_corpus(data_path: Path, web_path: Path) -> str:
    pieces = [
        "Русскоязычный текстовый корпус для предобучения SWIN AI. Задача модели — предсказывать следующий токен по предыдущему контексту.",
    ]
    if data_path.is_file():
        pieces.append(data_path.read_text(encoding="utf-8"))
    if web_path.is_file():
        pieces.append(web_path.read_text(encoding="utf-8"))
    return join_corpus(*pieces)


def build_instruction_corpus(
    knowledge_path: Path,
    dialogue_path: Path,
    instructions_path: Path,
) -> str:
    pieces = [
        "Ты — SWIN AI, самостоятельный русскоязычный помощник по программированию, разработке игр и техническим вопросам.",
        "Отвечай прямо, спокойно и по существу. Не выдумывай факты. Если данных недостаточно, честно обозначь это и предложи проверяемый следующий шаг.",
    ]
    if knowledge_path.is_file():
        entries = json.loads(knowledge_path.read_text(encoding="utf-8"))
        for item in entries:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            title = str(item.get("title", "")).strip()
            if question and answer:
                pieces.append(f"Тема: {title}\nПользователь: {question}\nSWIN: {answer}")
    pieces.extend(read_instruction_file(dialogue_path))
    pieces.extend(read_instruction_file(instructions_path))
    return join_corpus(*pieces)


def build_tensor_splits(train_corpus: str, validation_corpus: str, block_size: int):
    train_encoded = ByteTokenizer.encode(train_corpus)
    if len(train_encoded) < block_size * 4:
        raise ValueError("Корпус этапа обучения слишком мал. Добавь больше проверенных данных.")
    if len(set(train_encoded)) < 20:
        raise ValueError("В корпусе слишком мало разных байтов.")

    if validation_corpus.strip():
        validation_encoded = ByteTokenizer.encode(validation_corpus)
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
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, steps=8):
    model.eval()
    result = {}
    for name, data in (("train", train_data), ("validation", val_data)):
        losses = []
        for _ in range(steps):
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
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    batches_per_epoch = max(
        10,
        min(80, math.ceil(len(train_data) / (batch_size * block_size)) * 2),
    )
    best_validation = float("inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for _ in range(batches_per_epoch):
            x, y = sample_batch(train_data, batch_size, block_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        metrics = estimate_loss(
            model, train_data, val_data, batch_size, block_size, device
        )
        print(
            f"{stage_name} epoch {epoch}/{epochs} "
            f"train_loss={total_loss / batches_per_epoch:.4f} "
            f"validation_loss={metrics['validation']:.4f}"
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
    parser.add_argument("--instructions-train", default="data/openassistant_ru_train.jsonl")
    parser.add_argument("--instructions-validation", default="data/openassistant_ru_validation.jsonl")
    parser.add_argument("--output", default="models/swin-transformer.pt")
    parser.add_argument("--pretrain-output", default="models/swin-transformer-pretrain.pt")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--sft-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not 1 <= args.pretrain_epochs <= 200 or not 1 <= args.sft_epochs <= 100:
        parser.error("Недопустимое число эпох.")
    if not 4 <= args.batch_size <= 128:
        parser.error("--batch-size должен быть от 4 до 128")
    if args.d_model % args.heads != 0 or args.block_size < 32:
        parser.error("Проверь --d-model, --heads и --block-size.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство обучения: {device}")

    # Этап 1: языковое предобучение на очищенных русских веб-документах.
    pretrain_train = build_pretraining_corpus(Path(args.data), Path(args.web_train))
    pretrain_validation = (
        Path(args.web_validation).read_text(encoding="utf-8")
        if Path(args.web_validation).is_file()
        else ""
    )
    pretrain_train_data, pretrain_val_data = build_tensor_splits(
        pretrain_train, pretrain_validation, args.block_size
    )
    print(
        f"Этап 1 / pretraining: train={pretrain_train_data.numel():,} байт, "
        f"validation={pretrain_val_data.numel():,} байт"
    )

    model = SWINTransformer(
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=0.1,
    )
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

    # Этап 2: instruction tuning на парах вопрос/ответ, отдельно от веб-текста.
    sft_train = build_instruction_corpus(
        Path(args.knowledge),
        Path(args.dialogue),
        Path(args.instructions_train),
    )
    sft_validation = (
        join_corpus(*read_instruction_file(Path(args.instructions_validation)))
        if Path(args.instructions_validation).is_file()
        else ""
    )
    sft_train_data, sft_val_data = build_tensor_splits(
        sft_train, sft_validation, args.block_size
    )
    print(
        f"Этап 2 / instruction tuning: train={sft_train_data.numel():,} байт, "
        f"validation={sft_val_data.numel():,} байт"
    )
    model = train_stage(
        model,
        sft_train_data,
        sft_val_data,
        args.sft_epochs,
        args.batch_size,
        args.block_size,
        device,
        Path(args.output),
        "INSTRUCTION",
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
