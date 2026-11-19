"""Local SWIN agent orchestration: deterministic planning, tools and evidence.

This module intentionally contains no remote model/provider integration. It plans
bounded work from the user's request, invokes only registered local tools, and
returns a structured evidence ledger with explicit unknowns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AgentStep:
    name: str
    purpose: str
    tool: str
    required: bool = True


LOCAL_TOOL_NAMES = frozenset({
    "knowledge.search",
    "memory.select",
    "code.detect_language",
    "code.validate",
    "project.validate_manifest",
})

_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_local_tool(name: str, function: Callable[..., Any]) -> None:
    """Register a local, in-process tool using an explicit stable name."""
    if name not in LOCAL_TOOL_NAMES:
        raise ValueError(f"Unapproved local tool: {name}")
    if not callable(function):
        raise TypeError("A local tool must be callable.")
    _TOOL_REGISTRY[name] = function


def unregister_local_tool(name: str) -> None:
    _TOOL_REGISTRY.pop(name, None)


def registered_local_tools() -> list[str]:
    return sorted(_TOOL_REGISTRY)


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def plan_task(message: str, language: str = "ru") -> list[AgentStep]:
    """Select a small deterministic workflow; do not invent model reasoning."""
    query = re.sub(r"\s+", " ", str(message or "")).strip()
    if not query:
        return []
    normalized = query.casefold()
    steps: list[AgentStep] = [
        AgentStep("understand_request", "Выделить цель, ограничения и формат результата", "planner.parse"),
    ]
    project_markers = (
        "создай проект", "сделай проект", "сгенерируй проект", "полный проект",
        "create project", "build a project", "make an app", "создай приложение",
        "создай игру", "сделай приложение", "напиши программу целиком",
    )
    coding_markers = (
        "код", "скрипт", "функци", "класс", "программ", "ошибк", "исправ",
        "python", "godot", "gdscript", "kotlin", "javascript", "typescript",
        "code", "script", "function", "debug", "compile", "build",
    )
    knowledge_markers = (
        "что такое", "объясни", "расскажи", "почему", "как работает", "определение",
        "объяснить", "what is", "explain", "how does", "why ",
    )

    if _contains_any(normalized, project_markers):
        steps.extend([
            AgentStep("inspect_context", "Учесть память и существующий контекст проекта", "memory.select"),
            AgentStep("design_project", "Определить файлы, связи и точку входа", "project.design"),
            AgentStep("validate_project", "Проверить манифест, пути и синтаксис", "project.validate_manifest"),
        ])
    elif _contains_any(normalized, coding_markers):
        steps.extend([
            AgentStep("inspect_context", "Учесть выбранный язык и контекст проекта", "memory.select"),
            AgentStep("identify_language", "Определить язык и целевую платформу", "code.detect_language"),
            AgentStep("validate_code", "Проверить синтаксис без исполнения исходников", "code.validate"),
        ])
    elif _contains_any(normalized, knowledge_markers) or len(normalized.split()) >= 4:
        steps.extend([
            AgentStep("retrieve_knowledge", "Найти релевантные локальные фрагменты и источники", "knowledge.search"),
            AgentStep("select_memory", "Добавить только относящиеся к запросу записи памяти", "memory.select"),
            AgentStep("synthesize_evidence", "Сформировать ответ по найденным данным и отметить пробелы", "reasoning.local_synthesis"),
        ])
    else:
        steps.append(
            AgentStep("respond", "Подготовить ответ без обращения к внешним провайдерам", "response.local_fallback")
        )
    steps.append(
        AgentStep("report_limits", "Отделить подтверждённые сведения от предположений и неизвестного", "evidence.report")
    )
    return steps[:8]


def build_evidence_ledger(
    *,
    facts: list[dict[str, Any]] | None = None,
    inferences: list[dict[str, Any]] | None = None,
    unknowns: list[str] | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Preserve provenance and confidence categories instead of overclaiming."""
    normalized_facts = []
    for item in facts or []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", "")).strip()
        if value:
            normalized_facts.append({
                "value": value[:1200],
                "source": str(item.get("source", "local_tool"))[:160],
                "confidence": max(0.0, min(1.0, float(item.get("confidence", 1.0)))),
            })
    normalized_inferences = []
    for item in inferences or []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", "")).strip()
        if value:
            normalized_inferences.append({
                "value": value[:1200],
                "basis": str(item.get("basis", ""))[:500],
                "confidence": max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
            })
    normalized_sources = []
    seen_sources = set()
    for item in sources or []:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        key = (uri, title)
        if key in seen_sources:
            continue
        seen_sources.add(key)
        normalized_sources.append({
            "title": title[:180],
            "url": uri[:800],
            "source_type": str(item.get("source_type", "local"))[:80],
        })
    return {
        "facts": normalized_facts[:20],
        "inferences": normalized_inferences[:20],
        "unknowns": [str(item).strip()[:500] for item in (unknowns or []) if str(item).strip()][:20],
        "sources": normalized_sources[:12],
        "execution_performed": False,
    }


def execute_local_plan(
    message: str,
    *,
    language: str = "ru",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only registered in-process tools; report unavailable steps explicitly."""
    payload = dict(context or {})
    payload["message"] = str(message or "")
    payload["language"] = language
    steps = plan_task(message, language)
    outcomes: list[dict[str, Any]] = []
    for step in steps:
        tool = _TOOL_REGISTRY.get(step.tool)
        if tool is None:
            outcomes.append({
                "step": step.name,
                "tool": step.tool,
                "status": "planned" if step.tool in {
                    "planner.parse", "project.design", "reasoning.local_synthesis",
                    "evidence.report", "response.local_fallback",
                } else "unavailable",
            })
            continue
        try:
            result = tool(payload)
            outcomes.append({
                "step": step.name,
                "tool": step.tool,
                "status": "completed",
                "result": result,
            })
        except Exception as error:
            outcomes.append({
                "step": step.name,
                "tool": step.tool,
                "status": "failed",
                "error": str(error)[:300],
            })
    return {
        "engine": "swin-local-agent",
        "steps": [
            {"name": step.name, "purpose": step.purpose, "tool": step.tool}
            for step in steps
        ],
        "tool_results": outcomes,
    }


def capability_report() -> dict[str, Any]:
    """Describe the local engine, model status and capability limits honestly."""
    return {
        "engine": "swin-local-agent",
        "provider_dependency": False,
        "external_model_api": False,
        "tools": registered_local_tools(),
        "workflow_planner": True,
        "evidence_ledger": True,
        "knowledge_retrieval": True,
        "code_static_validation": True,
        "multi_file_project_scaffolding": True,
        "multi_file_llm_generation": False,
        "reasoning_weights": "experimental_local_checkpoint_not_quality_gated",
        "vision": {
            "enabled": False,
            "reason": "В локальной сборке нет установленной мультимодальной модели.",
            "metadata_only_fallback": True,
        },
    }
