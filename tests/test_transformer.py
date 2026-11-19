import tempfile
import unittest
from pathlib import Path

import torch

from transformer import ByteTokenizer, CharTokenizer, SWINTransformer, load_transformer, save_transformer


class ByteTokenizerTests(unittest.TestCase):
    def test_russian_utf8_round_trip(self):
        text = "Привет, SWIN AI! 🌍"
        encoded = ByteTokenizer.encode(text)
        self.assertTrue(all(0 <= token <= 255 for token in encoded))
        self.assertEqual(ByteTokenizer.decode(encoded), text)


class CharTokenizerTests(unittest.TestCase):
    def test_russian_and_code_round_trip_as_single_characters(self):
        text = "Привет, SWIN AI!\\nprint(42)"
        tokenizer = CharTokenizer.from_texts([text])
        encoded = tokenizer.encode(text)
        self.assertEqual(len(encoded), len(text))
        self.assertEqual(tokenizer.decode(encoded), text)

    def test_unknown_character_is_explicit(self):
        tokenizer = CharTokenizer.from_texts(["abc"])
        decoded = tokenizer.decode(tokenizer.encode("a🌍"))
        self.assertEqual(decoded, "a�")


class SWINTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = SWINTransformer(
            block_size=24,
            d_model=32,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
        )
        self.model.eval()

    def test_kv_cache_incremental_logits_match_full_forward(self):
        tokens = torch.randint(0, 256, (1, 14))
        with torch.inference_mode():
            full_logits, _ = self.model(tokens)
            cached_logits, caches = self.model.forward_with_cache(tokens[:, :5])
            for position in range(5, tokens.shape[1]):
                token = tokens[:, position:position + 1]
                cached_logits, caches = self.model.forward_incremental(
                    token, position, caches
                )
                self.assertTrue(
                    torch.allclose(
                        cached_logits[:, 0, :],
                        full_logits[:, position, :],
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )

    def test_old_torch_transformer_weights_load_without_translation(self):
        reference_layer = torch.nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        reference_encoder = torch.nn.TransformerEncoder(
            reference_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.model.blocks.load_state_dict(reference_encoder.state_dict(), strict=True)

    def test_greedy_generation_is_deterministic(self):
        first = self.model.generate("Пользователь:", max_new_tokens=12, temperature=0)
        second = self.model.generate("Пользователь:", max_new_tokens=12, temperature=0)
        self.assertEqual(first, second)

    def test_forward_and_loss_shapes(self):
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        logits, loss = self.model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 16, 256))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_future_tokens_do_not_change_previous_logits(self):
        first = torch.randint(0, 256, (1, 16))
        second = first.clone()
        second[0, 12:] = (second[0, 12:] + 17) % 256
        with torch.no_grad():
            logits_a, _ = self.model(first)
            logits_b, _ = self.model(second)
        self.assertTrue(torch.allclose(logits_a[:, :12], logits_b[:, :12], atol=1e-5))

    def test_generate_returns_valid_text(self):
        result = self.model.generate("Пользователь:", max_new_tokens=8)
        self.assertIsInstance(result, str)

    def test_character_checkpoint_round_trip_preserves_vocabulary(self):
        tokenizer = CharTokenizer.from_texts(["Пользователь: Привет\\nSWIN: Здравствуйте!"])
        model = SWINTransformer(
            block_size=32,
            d_model=32,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
            vocab_size=tokenizer.vocab_size,
        )
        model.tokenizer = tokenizer
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "swin-char.pt"
            save_transformer(model, str(path), validation_loss=1.0)
            loaded = load_transformer(str(path))
            self.assertIsInstance(loaded.tokenizer, CharTokenizer)
            self.assertEqual(loaded.tokenizer.vocabulary, tokenizer.vocabulary)
            tokens = torch.tensor([tokenizer.encode("Привет")], dtype=torch.long)
            with torch.no_grad():
                before, _ = model(tokens)
                after, _ = loaded(tokens)
            self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "swin.pt"
            save_transformer(self.model, str(path), validation_loss=1.25)
            loaded = load_transformer(str(path))
            self.assertEqual(loaded.block_size, self.model.block_size)
            sample = torch.randint(0, 256, (1, 12))
            with torch.no_grad():
                before, _ = self.model(sample)
                after, _ = loaded(sample)
            self.assertTrue(torch.allclose(before, after, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
