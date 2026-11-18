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


class SWINTransformer(nn.Module):
    """Small decoder-only Transformer trained from scratch on next-byte prediction."""

    def __init__(
        self,
        block_size: int = 128,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
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

        self.token_embedding = nn.Embedding(BYTE_VOCAB_SIZE, self.d_model)
        self.position_embedding = nn.Embedding(self.block_size, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=self.n_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, BYTE_VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            raise ValueError("Последовательность длиннее block_size.")

        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.blocks(x, mask=causal_mask)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, BYTE_VOCAB_SIZE),
                targets.reshape(batch_size * sequence_length),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 240,
        temperature: float = 0.75,
        top_k: int = 40,
        stop_at_newline: bool = True,
    ) -> str:
        self.eval()
        tokenizer = ByteTokenizer()
        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [ord(" ")]
        generated: list[int] = []

        for _ in range(max(0, max_new_tokens)):
            context = torch.tensor(
                [token_ids[-self.block_size:]],
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            logits, _ = self(context)
            next_logits = logits[0, -1, :] / max(float(temperature), 0.05)

            if 1 <= top_k < BYTE_VOCAB_SIZE:
                threshold = torch.topk(next_logits, top_k).values[-1]
                next_logits[next_logits < threshold] = float("-inf")

            probabilities = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
            generated.append(next_id)
            token_ids.append(next_id)

            if stop_at_newline and next_id == 10:
                break

        return tokenizer.decode(generated)


def save_transformer(model: SWINTransformer, path: str, validation_loss: float | None = None):
    checkpoint = {
        "format_version": 1,
        "tokenizer": "utf8_bytes",
        "config": {
            "block_size": model.block_size,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "n_layers": model.n_layers,
        },
        "state_dict": model.state_dict(),
        "validation_loss": validation_loss,
    }
    torch.save(checkpoint, path)


def load_transformer(path: str) -> SWINTransformer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("tokenizer") != "utf8_bytes":
        raise ValueError("Неподдерживаемый формат файла Transformer.")
    model = SWINTransformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model
