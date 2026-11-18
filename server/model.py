import numpy as np


class TinyCharRNN:
    """Учебная символьная RNN: предсказывает следующий символ, а не смысл."""

    def __init__(self, chars, hidden=96, seed=7):
        self.chars = list(chars)
        if not self.chars:
            raise ValueError("Словарь символов пуст.")
        self.stoi = {char: i for i, char in enumerate(self.chars)}
        self.itos = {i: char for char, i in self.stoi.items()}
        self.vocab = len(self.chars)
        self.hidden = int(hidden)

        rng = np.random.default_rng(seed)
        scale = 0.08
        self.Wxh = rng.normal(0, scale, (self.hidden, self.vocab)).astype(np.float32)
        self.Whh = rng.normal(0, scale, (self.hidden, self.hidden)).astype(np.float32)
        self.Why = rng.normal(0, scale, (self.vocab, self.hidden)).astype(np.float32)
        self.bh = np.zeros((self.hidden, 1), np.float32)
        self.by = np.zeros((self.vocab, 1), np.float32)

    def train_text(self, text, epochs=20, seq_len=48, lr=0.06):
        ids = [self.stoi[ch] for ch in text if ch in self.stoi]
        if len(ids) < seq_len + 2:
            raise ValueError("Мало данных: нужен текст длиннее seq_len + 2.")
        if epochs < 1:
            raise ValueError("Число эпох должно быть положительным.")

        for epoch in range(epochs):
            h_prev = np.zeros((self.hidden, 1), np.float32)
            total_loss = 0.0
            tokens_seen = 0

            for pos in range(0, len(ids) - seq_len - 1, seq_len):
                xs = ids[pos:pos + seq_len]
                ys = ids[pos + 1:pos + seq_len + 1]
                hs = {-1: h_prev}
                probs = {}

                for t, char_id in enumerate(xs):
                    x_vec = np.zeros((self.vocab, 1), np.float32)
                    x_vec[char_id] = 1.0
                    hs[t] = np.tanh(self.Wxh @ x_vec + self.Whh @ hs[t - 1] + self.bh)
                    logits = self.Why @ hs[t] + self.by
                    logits -= logits.max()
                    exp_logits = np.exp(logits)
                    probs[t] = exp_logits / exp_logits.sum()
                    total_loss -= float(np.log(probs[t][ys[t], 0] + 1e-12))

                tokens_seen += len(xs)
                d_wxh = np.zeros_like(self.Wxh)
                d_whh = np.zeros_like(self.Whh)
                d_why = np.zeros_like(self.Why)
                d_bh = np.zeros_like(self.bh)
                d_by = np.zeros_like(self.by)
                dh_next = np.zeros_like(h_prev)

                for t in reversed(range(len(xs))):
                    dy = probs[t].copy()
                    dy[ys[t]] -= 1.0
                    d_why += dy @ hs[t].T
                    d_by += dy
                    dh = self.Why.T @ dy + dh_next
                    raw = (1.0 - hs[t] * hs[t]) * dh
                    d_bh += raw
                    x_vec = np.zeros((self.vocab, 1), np.float32)
                    x_vec[xs[t]] = 1.0
                    d_wxh += raw @ x_vec.T
                    d_whh += raw @ hs[t - 1].T
                    dh_next = self.Whh.T @ raw

                # Средний градиент по последовательности уменьшает зависимость
                # шага обучения от длины окна.
                for grad in (d_wxh, d_whh, d_why, d_bh, d_by):
                    grad /= float(len(xs))
                    np.clip(grad, -5.0, 5.0, out=grad)

                self.Wxh -= lr * d_wxh
                self.Whh -= lr * d_whh
                self.Why -= lr * d_why
                self.bh -= lr * d_bh
                self.by -= lr * d_by
                h_prev = hs[len(xs) - 1]

            mean_loss = total_loss / max(1, tokens_seen)
            print(f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}")

    def generate(self, prompt="", length=180, temperature=0.8, seed=11):
        if length < 1:
            return prompt
        temperature = max(0.05, float(temperature))
        hidden = np.zeros((self.hidden, 1), np.float32)
        seed_text = prompt or self.chars[0]

        for char in seed_text:
            x_vec = np.zeros((self.vocab, 1), np.float32)
            x_vec[self.stoi.get(char, 0)] = 1.0
            hidden = np.tanh(self.Wxh @ x_vec + self.Whh @ hidden + self.bh)

        char_id = self.stoi.get(seed_text[-1], 0)
        rng = np.random.default_rng(seed)
        output = prompt

        for _ in range(length):
            x_vec = np.zeros((self.vocab, 1), np.float32)
            x_vec[char_id] = 1.0
            hidden = np.tanh(self.Wxh @ x_vec + self.Whh @ hidden + self.bh)
            logits = (self.Why @ hidden + self.by) / temperature
            logits -= logits.max()
            exp_logits = np.exp(logits)
            probabilities = exp_logits / exp_logits.sum()
            char_id = int(rng.choice(self.vocab, p=probabilities[:, 0]))
            output += self.itos[char_id]
        return output

    def save(self, path):
        np.savez_compressed(
            path,
            chars=np.asarray(self.chars),
            hidden=self.hidden,
            Wxh=self.Wxh,
            Whh=self.Whh,
            Why=self.Why,
            bh=self.bh,
            by=self.by,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            obj = cls(data["chars"].tolist(), int(data["hidden"]))
            for name in ("Wxh", "Whh", "Why", "bh", "by"):
                value = data[name].astype(np.float32)
                if value.shape != getattr(obj, name).shape:
                    raise ValueError(f"Неверная форма параметра {name}")
                setattr(obj, name, value)
        return obj
