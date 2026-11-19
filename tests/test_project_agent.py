import json
import unittest

from project_agent import ProjectBundleError, _safe_relative_path, build_project_prompt, generate_project_bundle, generate_local_project_scaffold, validate_project_manifest


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

    def test_local_python_scaffold_has_a_real_entry_point_and_test(self):
        result = generate_local_project_scaffold(
            "Создай приложение учёта задач на Python",
            "python",
        )
        paths = {item["path"] for item in result["files"]}
        self.assertEqual(result["scope"], "template_only")
        self.assertEqual(result["status"], "starter_scaffold")
        self.assertIn("main.py", paths)
        self.assertIn("src/app.py", paths)
        self.assertIn("tests/test_app.py", paths)
        self.assertIn("README.md", paths)
        self.assertTrue(all(item["content"] for item in result["files"] if item["path"] != "src/__init__.py"))

    def test_local_godot_scaffold_preserves_scene_and_script_folders(self):
        result = generate_local_project_scaffold(
            "Создай игру на Godot 4 с главной сценой",
            "gdscript",
        )
        paths = {item["path"] for item in result["files"]}
        self.assertIn("project.godot", paths)
        self.assertIn("scenes/main.tscn", paths)
        self.assertIn("scripts/main.gd", paths)
        self.assertEqual(result["generation_engine"] if "generation_engine" in result else "swin_local_templates", "swin_local_templates")

    def test_local_web_scaffold_uses_no_remote_runtime_api(self):
        result = generate_local_project_scaffold(
            "Создай сайт с HTML, CSS и JavaScript",
            "html",
        )
        paths = {item["path"] for item in result["files"]}
        self.assertTrue({"index.html", "styles.css", "app.js"}.issubset(paths))
        html = next(item["content"] for item in result["files"] if item["path"] == "index.html")
        self.assertNotIn("https://", html)
        self.assertIn("template_only", result["scope"])

    def test_local_scaffold_rejects_too_small_file_limit(self):
        with self.assertRaisesRegex(ProjectBundleError, "нужно не менее"):
            generate_local_project_scaffold(
                "Создай программу на Python",
                "python",
                max_files=2,
            )

    def test_project_manifest_validator_checks_every_file(self):
        result = validate_project_manifest([
            {"path": "src/main.py", "language": "python", "content": "def main():\\n    return 1\\n"},
            {"path": "README.md", "language": "other", "content": "# Demo\\n"},
        ])
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(result["failed_files"], 0)
        self.assertEqual(result["runtime"], "not_run")
        self.assertFalse(result["execution_performed"])

    def test_project_manifest_validator_rejects_unsafe_and_duplicate_paths(self):
        with self.assertRaises(ProjectBundleError):
            validate_project_manifest([{"path": "../escape.py", "content": "print(1)"}])
        with self.assertRaisesRegex(ProjectBundleError, "Повторяется путь"):
            validate_project_manifest([
                {"path": "src/main.py", "content": "print(1)"},
                {"path": "SRC/MAIN.py", "content": "print(2)"},
            ])

    def test_project_manifest_validator_marks_invalid_python(self):
        result = validate_project_manifest([
            {"path": "main.py", "content": "def broken(:\\n"}
        ])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["failed_files"], 1)

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
