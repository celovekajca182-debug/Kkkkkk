import argparse
from pathlib import Path

from model import TinyCharRNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SWIN AI character model from a local text corpus.")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--output", default="model.npz")
    args = parser.parse_args()

    if not 1 <= args.epochs <= 500:
        parser.error("--epochs должен быть в диапазоне от 1 до 500")
    if not 16 <= args.hidden <= 512:
        parser.error("--hidden должен быть в диапазоне от 16 до 512")

    data_path = Path(args.data)
    if not data_path.is_file():
        parser.error(f"Файл корпуса не найден: {data_path}")

    corpus = data_path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    if len(corpus) < 100:
        parser.error("Корпус слишком короткий: добавь не менее 100 символов текста.")
    unique_chars = len(set(corpus))
    if unique_chars < 12:
        parser.error("В корпусе слишком мало разных символов для обучения.")

    print(f"Корпус: {len(corpus)} символов; алфавит: {unique_chars} символов")
    print("Важно: это учебная символьная RNN, не большая диалоговая модель.")
    model = TinyCharRNN(sorted(set(corpus)), args.hidden)
    model.train_text(corpus, args.epochs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"Веса сохранены: {output_path}")


if __name__ == "__main__":
    main()
