import tempfile
import unittest
from pathlib import Path

from model import TinyCharRNN


class TinyCharRNNTests(unittest.TestCase):
    def test_train_generate_and_save_reload(self):
        corpus = ("SWIN AI учится на собственном тексте. " * 20)
        model = TinyCharRNN(sorted(set(corpus)), hidden=24, seed=3)
        model.train_text(corpus, epochs=1, seq_len=16, lr=0.02)

        prompt = "SWIN"
        generated = model.generate(prompt, length=12, seed=4)
        self.assertTrue(generated.startswith(prompt))
        self.assertGreaterEqual(len(generated), len(prompt) + 12)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights.npz"
            model.save(str(path))
            loaded = TinyCharRNN.load(str(path))
            self.assertEqual(loaded.vocab, model.vocab)
            self.assertEqual(loaded.hidden, model.hidden)


if __name__ == "__main__":
    unittest.main()
