import tempfile
import unittest
from pathlib import Path

import torch

from transformer import ByteTokenizer, SWINTransformer, load_transformer, save_transformer


class ByteTokenizerTests(unittest.TestCase):
    def test_russian_utf8_round_trip(self):
        text = "Привет, SWIN AI! 🌍"
        encoded = ByteTokenizer.encode(text)
        self.assertTrue(all(0 <= token <= 255 for token in encoded))
        self.assertEqual(ByteTokenizer.decode(encoded), text)


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
