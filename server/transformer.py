import torch
from torch import nn
from torch.nn import functional as F

BYTE_VOCAB_SIZE = 256


class ByteTokenizer:
    """UTF-8 byte tokenizer: supports Russian and arbitrary Unicode without a vocab file."""

    vocab_size = BYTE_VOCAB_SIZE

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return bytes(int(token_id) for token_id in token_ids).decode("utf-8", errors="replace")


class CharTokenizer:
    """Unicode character tokenizer: Russian letters and code symbols are one token each."""
    UNK = "<UNK>"

    def __init__(self, vocabulary: list[str]):
        unique = list(dict.fromkeys(vocabulary))
        if not unique or unique[0] != self.UNK:
            unique = [self.UNK] + [char for char in unique if char != self.UNK]
        self.vocabulary = unique
        self.stoi = {char: index for index, char in enumerate(unique)}
        self.vocab_size = len(unique)

    @classmethod
    def from_texts(cls, texts, max_vocab: int = 4096):
        from collections import Counter
        counts = Counter(
            char for text in texts
            for char in text
            if char.isprintable() or char in "\n\r\t"
        )
        frequent = [char for char, _ in counts.most_common(max(1, max_vocab - 1))]
        return cls([cls.UNK] + sorted(char for char in frequent if char != cls.UNK))

    def encode(self, text: str) -> list[int]:
        unknown_id = self.stoi[self.UNK]
        return [self.stoi.get(char, unknown_id) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        result = []
        for token_id in token_ids:
            index = int(token_id)
            if 0 <= index < self.vocab_size:
                char = self.vocabulary[index]
                result.append("\ufffd" if char == self.UNK else char)
            else:
                result.append("\ufffd")
        return "".join(result)


class SWINMultiheadAttention(nn.Module):
    """Multi-head causal attention with incremental key/value caching.

    Parameter names intentionally match torch.nn.MultiheadAttention so previously
    trained SWIN checkpoints can be loaded without translating their state_dict.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model должен делиться на n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)

        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)

    def _project(self, x: torch.Tensor):
        batch_size, sequence_length, _ = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        def split_heads(value):
            return value.view(
                batch_size, sequence_length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        return split_heads(q), split_heads(k), split_heads(v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project(x)
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        merged = attention.transpose(1, 2).contiguous().view(
            x.shape[0], x.shape[1], self.d_model
        )
        return self.out_proj(merged)

    def forward_cached(self, x: torch.Tensor, cache=None):
        """Process a new chunk and return output plus concatenated (key, value)."""
        q, new_k, new_v = self._project(x)
        if cache is None:
            keys, values = new_k, new_v
            causal = True
        else:
            old_k, old_v = cache
            keys = torch.cat((old_k, new_k), dim=2)
            values = torch.cat((old_v, new_v), dim=2)
            causal = False

        attention = F.scaled_dot_product_attention(
            q,
            keys,
            values,
            dropout_p=0.0,
            is_causal=causal,
        )
        merged = attention.transpose(1, 2).contiguous().view(
            x.shape[0], x.shape[1], self.d_model
        )
        return self.out_proj(merged), (keys, values)


class SWINTransformerLayer(nn.Module):
    """Pre-norm Transformer block with state_dict-compatible parameter names."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = SWINMultiheadAttention(d_model, n_heads, dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.self_attn(self.norm1(x)))
        feed_forward = self.linear2(
            self.dropout(F.gelu(self.linear1(self.norm2(x))))
        )
        return x + self.dropout2(feed_forward)

    def forward_cached(self, x: torch.Tensor, cache=None):
        attention, new_cache = self.self_attn.forward_cached(self.norm1(x), cache)
        x = x + attention
        feed_forward = self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return x + feed_forward, new_cache


class SWINTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SWINTransformerLayer(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_cached(self, x: torch.Tensor, caches=None):
        if caches is None:
            caches = [None] * len(self.layers)
        new_caches = []
        for layer, cache in zip(self.layers, caches):
            x, new_cache = layer.forward_cached(x, cache)
            new_caches.append(new_cache)
        return x, new_caches


class SWINTransformer(nn.Module):
    """Small decoder-only Transformer trained from scratch on next-byte prediction."""

    def __init__(
        self,
        block_size: int = 128,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        vocab_size: int = BYTE_VOCAB_SIZE,
    ): 
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model должен делиться на n_heads.")
        if block_size < 8:
            raise ValueError("block_size должен быть не меньше 8.")

        self.block_size = int(block_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.vocab_size = int(vocab_size)
        if self.vocab_size < 2:
            raise ValueError("vocab_size должен быть не меньше 2.")
        self.tokenizer = ByteTokenizer()

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.block_size, self.d_model)
        self.register_buffer(
            "position_ids", torch.arange(self.block_size, dtype=torch.long), persistent=False
        )
        self.blocks = SWINTransformerEncoder(
            self.d_model, self.n_heads, self.n_layers, dropout
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(x))

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            raise ValueError("Последовательность длиннее block_size.")

        positions = self.position_ids[:sequence_length]
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x = self.blocks(x)
        logits = self._logits(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, self.vocab_size),
                targets.reshape(batch_size * sequence_length),
            )
        return logits, loss

    @torch.inference_mode()
    def forward_with_cache(self, token_ids: torch.Tensor):
        """Prefill prompt once and return the per-layer attention KV cache."""
        batch_size, sequence_length = token_ids.shape
        if sequence_length < 1 or sequence_length > self.block_size:
            raise ValueError("Размер prefill-последовательности вне допустимого диапазона.")
        positions = self.position_ids[:sequence_length]
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        x, caches = self.blocks.forward_cached(x)
        return self._logits(x), caches

    @torch.inference_mode()
    def forward_incremental(self, token_id: torch.Tensor, position: int, caches):
        """Process one new byte using cached keys and values."""
        if position < 0 or position >= self.block_size:
            raise ValueError("Позиция за пределами block_size.")
        if token_id.shape[1] != 1:
            raise ValueError("KV-cache принимает ровно один новый токен.")
        positions = self.position_ids[position:position + 1]
        x = self.token_embedding(token_id) + self.position_embedding(positions)[None, :, :]
        x, new_caches = self.blocks.forward_cached(x, caches)
        return self._logits(x), new_caches

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        top_k: int = 1,
        stop_at_newline: bool = True,
    ) -> str:
        self.eval()
        tokenizer = self.tokenizer
        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            token_ids = tokenizer.encode(" ") or [0]

        context = token_ids[-self.block_size:]
        context_tensor = torch.tensor(
            [context], dtype=torch.long, device=next(self.parameters()).device
        )
        logits, caches = self.forward_with_cache(context_tensor)
        generated: list[int] = []

        for _ in range(max(0, max_new_tokens)):
            next_logits = logits[0, -1, :]
            if temperature <= 0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / max(float(temperature), 0.05)
                if 1 <= top_k < self.vocab_size:
                    threshold = torch.topk(next_logits, top_k).values[-1]
                    next_logits[next_logits < threshold] = float("-inf")
                probabilities = F.softmax(next_logits, dim=-1)
                next_id = int(torch.multinomial(probabilities, num_samples=1).item())

            generated.append(next_id)
            token_ids.append(next_id)
            newline_id = tokenizer.stoi.get("\n", -1) if isinstance(tokenizer, CharTokenizer) else 10
            if stop_at_newline and next_id == newline_id:
                break

            if len(context) < self.block_size:
                new_token = torch.tensor(
                    [[next_id]], dtype=torch.long, device=context_tensor.device
                )
                logits, caches = self.forward_incremental(
                    new_token, position=len(context), caches=caches
                )
                context.append(next_id)
            else:
                # Rebase positional embeddings when the rolling context window is full.
                context = token_ids[-self.block_size:]
                context_tensor = torch.tensor(
                    [context], dtype=torch.long, device=context_tensor.device
                )
                logits, caches = self.forward_with_cache(context_tensor)

        return tokenizer.decode(generated)


def save_transformer(model: SWINTransformer, path: str, validation_loss: float | None = None):
    tokenizer_name = "unicode_char" if isinstance(model.tokenizer, CharTokenizer) else "utf8_bytes"
    checkpoint = {
        "format_version": 2 if tokenizer_name == "unicode_char" else 1,
        "tokenizer": tokenizer_name,
        "tokenizer_vocabulary": list(model.tokenizer.vocabulary) if isinstance(model.tokenizer, CharTokenizer) else None,
        "config": {
            "block_size": model.block_size,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "n_layers": model.n_layers,
            "vocab_size": model.vocab_size,
        },
        "state_dict": model.state_dict(),
        "validation_loss": validation_loss,
    }
    torch.save(checkpoint, path)


def load_transformer(path: str) -> SWINTransformer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer_name = checkpoint.get("tokenizer")
    config = dict(checkpoint.get("config", {}))
    if tokenizer_name == "utf8_bytes" and checkpoint.get("format_version") == 1:
        config.setdefault("vocab_size", BYTE_VOCAB_SIZE)
        tokenizer = ByteTokenizer()
    elif tokenizer_name == "unicode_char" and checkpoint.get("format_version") == 2:
        vocabulary = checkpoint.get("tokenizer_vocabulary")
        if not isinstance(vocabulary, list) or len(vocabulary) < 2:
            raise ValueError("В checkpoint отсутствует символьный словарь.")
        tokenizer = CharTokenizer(vocabulary)
        config["vocab_size"] = tokenizer.vocab_size
    else:
        raise ValueError("Неподдерживаемый формат файла Transformer.")
    model = SWINTransformer(**config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.tokenizer = tokenizer
    model.eval()
    return model
