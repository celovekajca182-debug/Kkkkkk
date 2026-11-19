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


VALID_MULTILINGUAL_OUTPUTS = [
    "Функция принимает аргументы, выполняет блок кода и возвращает результат. Она помогает разделять программу на небольшие части.",
    "Сначала прочитай полный Traceback, найди последнюю строку и определи тип ошибки. Затем проверь значение переменной и место вызова функции.",
    "A function is a named block of code that takes inputs and returns a result. It can be reused in different parts of a program.",
    "Eine Funktion ist ein benannter Codeblock. Sie nimmt Eingaben entgegen und gibt ein Ergebnis zurück. So kann man Code wiederverwenden.",
    "Una función es un bloque de código que recibe datos y devuelve un resultado. Puede reutilizarse en distintas partes del programa.",
    "Une fonction est un bloc de code qui reçoit des valeurs et renvoie un résultat. Elle peut être réutilisée dans un programme.",
    "函数是一段代码，可以接收参数并执行操作，然后返回计算结果。函数可以在程序的不同位置重复使用。",
    "Функція — це блок коду, який приймає аргументи та повертає результат. Її можна використовувати повторно.",
    "Uma função é um bloco de código que recebe valores e retorna um resultado. Ela pode ser reutilizada no programa.",
    "Una funzione è un blocco di codice che riceve valori e restituisce un risultato. Può essere riutilizzata nel programma.",
    "関数はコードのまとまりで、値を受け取り、処理を行い、結果を返します。プログラムの中で何度も使用できます。",
]

class EvaluateTransformerTests(unittest.TestCase):
    def test_repeated_byte_gibberish_fails(self):
        result = assess_generated_text("оооооооооооооооооооооо")
        self.assertEqual(result["status"], "failed")

    def test_invalid_utf8_replacement_fails(self):
        result = assess_generated_text("Это нормальный текст, но символ сломан � и дальше.")
        self.assertEqual(result["status"], "failed")

    def test_coherent_russian_paragraph_passes_heuristics(self):
        result = assess_generated_text(
            "Функция принимает аргументы, выполняет блок кода и возвращает результат. "
            "Проверь входные данные и обработай возможные ошибки."
        )
        self.assertEqual(result["status"], "passed")

    def test_chinese_and_japanese_text_pass_concept_checks(self):
        chinese = assess_generated_text(
            "函数是一段代码，可以接收参数并执行操作，然后返回计算结果。",
            required_concepts=[["函数"], ["代码"], ["结果", "返回"]],
        )
        japanese = assess_generated_text(
            "関数はコードのまとまりで、値を受け取り、結果を返します。",
            required_concepts=[["関数"], ["コード"], ["結果", "返"]],
        )
        self.assertEqual(chinese["status"], "passed")
        self.assertEqual(japanese["status"], "passed")

    def test_valid_python_function_passes(self):
        result = assess_generated_text(
            "def add(a, b):\n    return a + b\n",
            require_code=True,
            required_function="add",
        )
        self.assertEqual(result["status"], "passed")

    def test_valid_python_with_wrong_function_name_fails_required_probe(self):
        result = assess_generated_text(
            "def plus(a, b):\n    return a + b\n",
            require_code=True,
            required_function="add",
        )
        self.assertEqual(result["status"], "failed")

    def test_valid_python_with_wrong_operation_fails_required_probe(self):
        result = assess_generated_text(
            "def add(a, b):\n    return a - b\n",
            require_code=True,
            required_function="add",
        )
        self.assertEqual(result["status"], "failed")

    def test_russian_answer_must_cover_prompt_concepts(self):
        result = assess_generated_text(
            "Это простой и удобный инструмент для выполнения разных задач.",
            required_concepts=[["функц"], ["код"]],
        )
        self.assertEqual(result["status"], "failed")

    def test_python_code_without_valid_syntax_fails(self):
        result = assess_generated_text("def add(:\n    return 1", require_code=True)
        self.assertEqual(result["status"], "failed")

    def test_model_passes_only_when_text_and_code_probes_pass(self):
        model = FakeModel(VALID_MULTILINGUAL_OUTPUTS + ["def add(a, b):\n    return a + b\n"])
        report = evaluate_model(model)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["passed_text_probes"], 11)
        self.assertEqual(report["total_text_probes"], 11)
        self.assertEqual(len(set(report["passed_languages"])), 10)
        self.assertFalse(report["failed_languages"])
        self.assertTrue(report["python_code_probe_passed"])

    def test_model_is_rejected_when_any_required_probe_is_gibberish(self):
        outputs = list(VALID_MULTILINGUAL_OUTPUTS)
        outputs[3] = "оооооооооооооооооооо"
        model = FakeModel(outputs + ["def add(a, b):\n    return a + b\n"])
        report = evaluate_model(model)
        self.assertFalse(report["accepted"])
        self.assertIn("de", report["failed_languages"])


if __name__ == "__main__":
    unittest.main()
