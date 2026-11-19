import unittest

from local_agent import (
    build_evidence_ledger,
    capability_report,
    execute_local_plan,
    plan_task,
    register_local_tool,
    registered_local_tools,
    unregister_local_tool,
)


class LocalAgentTests(unittest.TestCase):
    def tearDown(self):
        for name in registered_local_tools():
            unregister_local_tool(name)

    def test_project_request_selects_local_project_workflow(self):
        steps = plan_task("Создай приложение для Android целиком")
        names = [step.name for step in steps]
        self.assertIn("design_project", names)
        self.assertIn("validate_project", names)
        self.assertEqual(steps[0].tool, "planner.parse")
        self.assertEqual(steps[-1].name, "report_limits")

    def test_code_request_selects_language_and_static_validation(self):
        steps = plan_task("Исправь ошибку в GDScript")
        self.assertIn("identify_language", [step.name for step in steps])
        self.assertIn("validate_code", [step.name for step in steps])

    def test_knowledge_request_selects_local_retrieval_and_memory(self):
        steps = plan_task("Объясни, как работает A* поиск пути")
        names = [step.name for step in steps]
        self.assertIn("retrieve_knowledge", names)
        self.assertIn("select_memory", names)

    def test_execute_local_plan_runs_only_registered_local_tools(self):
        calls = []
        register_local_tool(
            "code.validate",
            lambda context: calls.append(context.get("code")) or {"status": "passed"},
        )
        result = execute_local_plan(
            "Исправь функцию на Python",
            language="ru",
            context={"code": "def f(): pass"},
        )
        code_step = next(item for item in result["tool_results"] if item["tool"] == "code.validate")
        self.assertEqual(code_step["status"], "completed")
        self.assertEqual(code_step["result"]["status"], "passed")
        self.assertEqual(calls, ["def f(): pass"])
        self.assertEqual(result["engine"], "swin-local-agent")

    def test_execute_local_plan_reports_unavailable_tools_without_network_fallback(self):
        result = execute_local_plan("Создай проект приложения", language="ru")
        project_step = next(item for item in result["tool_results"] if item["tool"] == "project.design")
        self.assertEqual(project_step["status"], "planned")
        self.assertNotIn("network", str(result).lower())
        self.assertNotIn("provider", str(result).lower())

    def test_empty_request_has_no_steps(self):
        self.assertEqual(plan_task("   "), [])

    def test_only_approved_local_tools_can_be_registered(self):
        register_local_tool("knowledge.search", lambda query: [])
        self.assertEqual(registered_local_tools(), ["knowledge.search"])
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            register_local_tool("network.request", lambda: None)

    def test_evidence_ledger_keeps_facts_inferences_and_unknowns_separate(self):
        ledger = build_evidence_ledger(
            facts=[{"value": "Синтаксис Python проверен", "source": "python_ast", "confidence": 1.0}],
            inferences=[{"value": "Проект вероятно предназначен для Android", "basis": "Указана платформа Android", "confidence": 0.7}],
            unknowns=["Runtime не запускался"],
            sources=[{"title": "Python docs", "url": "https://docs.python.org/3/"}],
        )
        self.assertEqual(len(ledger["facts"]), 1)
        self.assertEqual(len(ledger["inferences"]), 1)
        self.assertEqual(ledger["unknowns"], ["Runtime не запускался"])
        self.assertFalse(ledger["execution_performed"])

    def test_capabilities_are_provider_free_and_honest(self):
        capabilities = capability_report()
        self.assertFalse(capabilities["provider_dependency"])
        self.assertFalse(capabilities["external_model_api"])
        self.assertFalse(capabilities["vision"]["enabled"])
        self.assertTrue(capabilities["multi_file_project_scaffolding"])
        self.assertFalse(capabilities["multi_file_llm_generation"])


if __name__ == "__main__":
    unittest.main()
