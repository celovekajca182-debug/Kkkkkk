import json
import unittest

from project_agent import ProjectBundleError, _safe_relative_path, build_project_prompt, generate_project_bundle


class ProjectAgentTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "project_name": "SWIN Demo",
            "summary": "Небольшой проект из нескольких файлов.",
            "files": [
                {"path": "main.py", "language": "python", "purpose": "Точка входа",
                 "content": "def greet(name: str) -> str:\n    return f'Привет, {name}!'\n"},
                {"path": "config.json", "language": "json", "purpose": "Конфигурация",
                 "content": '{"language": "ru", "enabled": true}'},
                {"path": "README.md", "language": "other", "purpose": "Инструкция",
                 "content": "# SWIN Demo\n\nЗапусти python main.py.\n"},
            ],
        }

    def test_generates_and_validates_a_multi_file_manifest(self):
        calls = []

        def generator(prompt, max_tokens):
            calls.append((prompt, max_tokens))
            return json.dumps(self.manifest, ensure_ascii=False)

        result = generate_project_bundle(
            "Создай небольшой Python проект с конфигурацией и README",
            "python",
            "ru",
            text_generator=generator,
        )
        self.assertEqual(result["status"], "generated_validated")
        self.assertEqual(result["file_count"], 3)
        self.assertEqual(result["project_name"], "SWIN Demo")
        self.assertEqual(result["files"][0]["validation"]["status"], "passed")
        self.assertEqual(result["files"][1]["validation"]["status"], "passed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 16000)
        self.assertIn("ONLY one valid JSON object", calls[0][0])

    def test_rejects_parent_traversal_and_absolute_paths(self):
        for path in ("../main.py", "/etc/passwd", "src/../../evil.py", "C:/temp.py"):
            with self.subTest(path=path):
                with self.assertRaises(ProjectBundleError):
                    _safe_relative_path(path)

    def test_rejects_duplicate_paths_case_insensitively(self):
        manifest = dict(self.manifest)
        manifest["files"] = list(self.manifest["files"]) + [
            {"path": "MAIN.py", "content": "print('duplicate')"}
        ]
        with self.assertRaisesRegex(ProjectBundleError, "Повторяется путь"):
            generate_project_bundle(
                "Создай проект", "python",
                text_generator=lambda prompt, count: json.dumps(manifest),
            )

    def test_rejects_too_many_files(self):
        manifest = dict(self.manifest)
        manifest["files"] = [
            {"path": f"file_{index}.txt", "content": "text"}
            for index in range(4)
        ]
        with self.assertRaisesRegex(ProjectBundleError, "превышает лимит"):
            generate_project_bundle(
                "Создай проект", "python", max_files=3,
                text_generator=lambda prompt, count: json.dumps(manifest),
            )

    def test_rejects_empty_or_invalid_json_manifest(self):
        with self.assertRaises(ProjectBundleError):
            generate_project_bundle(
                "Создай проект", "python",
                text_generator=lambda prompt, count: "not json",
            )

    def test_project_prompt_contains_safety_rules_and_context(self):
        prompt = build_project_prompt(
            "Сделай приложение", "python", "ru", "Используй SQLite"
        )
        self.assertIn("src/main.py", prompt)
        self.assertIn("Never use absolute paths", prompt)
        self.assertIn("Do not invent successful execution", prompt)
        self.assertIn("Используй SQLite", prompt)


if __name__ == "__main__":
    unittest.main()
