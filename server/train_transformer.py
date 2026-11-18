import argparse
import json
import math
import random
from pathlib import Path

import torch

from transformer import ByteTokenizer, SWINTransformer, save_transformer


def build_corpus(data_path: Path, knowledge_path: Path, dialogue_path: Path) -> str:
    pieces = [
        "Ты — SWIN AI, самостоятельный русскоязычный помощник по программированию, разработке игр и техническим вопросам.",
        "Отвечай прямо, спокойно и по существу. Не выдумывай факты. Если данных недостаточно, честно обозначь это и предложи проверяемый следующий шаг.",
        data_path.read_text(encoding="utf-8") if data_path.is_file() else "",
    ]

    if knowledge_path.is_file():
        entries = json.loads(knowledge_path.read_text(encoding="utf-8"))
        for item in entries:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            title = str(item.get("title", "")).strip()
            if question and answer:
                pieces.append(f"Тема: {title}\nПользователь: {question}\nSWIN: {answer}")

    if dialogue_path.is_file():
        for line_number, line in enumerate(dialogue_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Ошибка JSONL в строке {line_number}: {error}") from error
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question and answer:
                pieces.append(f"Пользователь: {question}\nSWIN: {answer}")

    return "\n\n".join(pieces).strip() + "\n"


def sample_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
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


def main():
    parser = argparse.ArgumentParser(description="Train SWIN AI's own byte-level Transformer weights.")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--knowledge", default="data/knowledge.json")
    parser.add_argument("--dialogue", default="data/dialogue.jsonl")
    parser.add_argument("--output", default="models/swin-transformer.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if not 1 <= args.epochs <= 200:
        parser.error("--epochs должен быть от 1 до 200")
    if not 4 <= args.batch_size <= 128:
        parser.error("--batch-size должен быть от 4 до 128")
    if args.d_model % args.heads != 0:
        parser.error("--d-model должен делиться на --heads")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство обучения: {device}")

    corpus = build_corpus(Path(args.data), Path(args.knowledge), Path(args.dialogue))
    encoded = ByteTokenizer.encode(corpus)
    if len(encoded) < args.block_size * 4:
        parser.error("Корпус слишком мал для выбранного block-size. Добавь проверенные примеры.")
    if len(set(encoded)) < 20:
        parser.error("В корпусе слишком мало разных байтов.")

    split = max(args.block_size + 2, int(len(encoded) * 0.9))
    split = min(split, len(encoded) - args.block_size - 2)
    train_data = torch.tensor(encoded[:split], dtype=torch.long)
    val_data = torch.tensor(encoded[split:], dtype=torch.long)
    print(f"Корпус: {len(encoded):,} UTF-8 байт; train={len(train_data):,}; validation={len(val_data):,}")
    print(f"Архитектура: byte Transformer, d_model={args.d_model}, heads={args.heads}, layers={args.layers}, block={args.block_size}")

    model = SWINTransformer(
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    batches_per_epoch = max(10, min(80, math.ceil(len(train_data) / (args.batch_size * args.block_size)) * 2))
    best_validation = float("inf")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for _ in range(batches_per_epoch):
            x, y = sample_batch(train_data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        metrics = estimate_loss(
            model, train_data, val_data, args.batch_size, args.block_size, device
        )
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={total_loss / batches_per_epoch:.4f} "
            f"validation_loss={metrics['validation']:.4f}"
        )

        if metrics["validation"] < best_validation:
            best_validation = metrics["validation"]
            model_cpu = model.to("cpu")
            save_transformer(model_cpu, str(output_path), best_validation)
            model.to(device)

    print(f"Лучшие validation loss: {best_validation:.4f}")
    print(f"Собственные веса Transformer сохранены: {output_path}")
    model = model.to("cpu").eval()
    sample = model.generate("Пользователь: Как исправить ошибку Godot?\nSWIN:", max_new_tokens=120)
    print("Пробная генерация:", sample[:300])


if __name__ == "__main__":
    main()
