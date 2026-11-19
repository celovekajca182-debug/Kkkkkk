import unittest

from evaluate_transformer import assess_generated_text, evaluate_model


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, prompt, **kwargs):
        value = self.outputs[self.calls]
        self.calls += 1
        return value


class EvaluateTransformerTests(unittest.TestCase):
    def test_repeated_byte_gibberish_fails(self):
        result = assess_generated_text("оооооооооооооооооооооо")
        self.assertEqual(result["status"], "failed")

    def test_invalid_utf8_replacement_fails(self):
        result = assess_generated_text("Это нормальный текст, но символ сломан � и дальше.")
        self.assertEqual(result["status"], "failed")

    def test_coherent_russian_paragraph_passes_heuristics(self):
        result = assess_generated_text(
            "Функция принимает аргументы, выполняет инструкции и возвращает результат. "
            "Проверь входные данные и обработай возможные ошибки."
        )
        self.assertEqual(result["status"], "passed")

    def test_valid_python_function_passes(self):
        result = assess_generated_text(
            "def add(a, b):\\n    return a + b\\n",
            require_code=True,
        )
        self.assertEqual(result["status"], "passed")

    def test_python_code_without_valid_syntax_fails(self):
        result = assess_generated_text("def add(:\\n    return 1", require_code=True)
        self.assertEqual(result["status"], "failed")

    def test_model_passes_only_when_text_and_code_probes_pass(self):
        model = FakeModel([
            "Функция получает значения, выполняет действия и возвращает вычисленный результат. "
            "Она помогает разделять программу на небольшие повторно используемые части.",
            "Сначала прочитай полный Traceback, найди последнюю строку и определи тип ошибки. "
            "Затем проверь значение переменной и место вызова функции.",
            "def add(a, b):\\n    return a + b\\n",
        ])
        report = evaluate_model(model)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["passed_text_probes"], 2)
        self.assertTrue(report["python_code_probe_passed"])

    def test_model_is_rejected_when_any_required_probe_is_gibberish(self):
        model = FakeModel([
            "оооооооооооооооооооо",
            "Сначала прочитай сообщение ошибки и проверь типы значений. "
            "Затем найди строку исходного кода, где возникло исключение.",
            "def add(a, b):\\n    return a + b\\n",
        ])
        report = evaluate_model(model)
        self.assertFalse(report["accepted"])


if __name__ == "__main__":
    unittest.main()
