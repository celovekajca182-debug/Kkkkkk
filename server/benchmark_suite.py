"""Live SWIN API benchmark: readiness, auth boundaries, and opt-in model quality probes.

Run the safe smoke checks:
    python server/benchmark_suite.py

Run two real inference probes (may consume provider credits):
    SWIN_BENCHMARK_TOKEN=<bearer-token> python server/benchmark_suite.py --live-model
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str = "",
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any], float]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read(1_000_000)
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read(1_000_000)
    elapsed_ms = (time.perf_counter() - started) * 1000
    try:
        data = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = {"raw_response": raw.decode("utf-8", "replace")[:500]}
    return status, data if isinstance(data, dict) else {"data": data}, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="SWIN API smoke and inference benchmark")
    parser.add_argument(
        "--base-url",
        default=os.getenv("SWIN_API_URL", "https://kkkkkk-02vl.onrender.com"),
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--live-model",
        action="store_true",
        help="Send one Qwen chat request and one code-generation request; provider credits may be consumed.",
    )
    args = parser.parse_args()
    token = os.getenv("SWIN_BENCHMARK_TOKEN", "").strip()
    if args.live_model and not token:
        parser.error("Set SWIN_BENCHMARK_TOKEN to an existing SWIN bearer token; do not pass it as a command-line argument.")

    results: list[tuple[str, bool, str, float | None]] = []

    def record(name: str, passed: bool, detail: str, elapsed: float | None = None) -> None:
        results.append((name, passed, detail, elapsed))
        marker = "PASS" if passed else "FAIL"
        timing = f" ({elapsed:.0f} ms)" if elapsed is not None else ""
        print(f"[{marker}] {name}{timing}: {detail}")

    status, health, latency = request_json(args.base_url, "/health", timeout=args.timeout)
    record("health_endpoint", status == 200 and health.get("ok") is True, f"HTTP {status}; ok={health.get('ok')}", latency)

    status, capabilities, latency = request_json(args.base_url, "/capabilities", timeout=args.timeout)
    record("capabilities_endpoint", status == 200, f"HTTP {status}", latency)
    reasoning = capabilities.get("reasoning", {}) if isinstance(capabilities, dict) else {}
    chat_ready = health.get("chat_provider_enabled") is True
    code_ready = health.get("code_provider_enabled") is True
    record("chat_model_configured", chat_ready, str(health.get("reasoning_model") or "not configured"))
    record("code_model_configured", code_ready, str(health.get("coding_model") or "not configured"))
    record(
        "capabilities_match_health",
        bool(reasoning.get("provider_enabled")) == bool(health.get("llm_provider_enabled")),
        f"capabilities={reasoning.get('provider_enabled')}; health={health.get('llm_provider_enabled')}",
    )
    record(
        "provider_key_not_exposed",
        "hf_" not in json.dumps(capabilities, ensure_ascii=False)
        and "api_key" not in json.dumps(capabilities, ensure_ascii=False).lower(),
        "Public capabilities response contains no provider credential fields.",
    )

    probes = [
        ("protected_memory", "GET", "/memory/facts", None),
        ("protected_chat", "POST", "/chat", {"message": "auth probe", "session_id": "auth-probe"}),
        ("protected_code", "POST", "/code/assist", {"task": "auth probe", "language": "python"}),
    ]
    for name, method, path, payload in probes:
        status, _, latency = request_json(
            args.base_url, path, method=method, payload=payload, timeout=args.timeout
        )
        record(name, status == 401, f"Unauthenticated request returned HTTP {status}, expected 401.", latency)

    if args.live_model:
        session_id = "benchmark-" + uuid.uuid4().hex[:16]
        try:
            status, chat, latency = request_json(
                args.base_url,
                "/chat",
                method="POST",
                token=token,
                timeout=args.timeout,
                payload={
                    "message": "Сколько будет 19 умножить на 23? Ответь кратко и обязательно укажи точное число.",
                    "session_id": session_id,
                    "response_language": "ru",
                },
            )
            answer = str(chat.get("answer", ""))
            record(
                "qwen_chat_arithmetic",
                status == 200 and bool(re.search(r"(?<!\\d)437(?!\\d)", answer)),
                f"HTTP {status}; answer={answer[:180]!r}",
                latency,
            )

            status, code, latency = request_json(
                args.base_url,
                "/code/assist",
                method="POST",
                token=token,
                timeout=max(args.timeout, 90.0),
                payload={
                    "task": "Напиши функцию Python add(a, b), которая возвращает сумму двух аргументов. Верни полный код.",
                    "language": "python",
                    "response_language": "ru",
                    "research_web": False,
                    "show_activity": False,
                },
            )
            validation = code.get("validation", {})
            source = str(code.get("code", ""))
            record(
                "qwen_code_python_ast",
                status == 200 and validation.get("status") == "passed" and "def " in source,
                f"HTTP {status}; status={code.get('status')}; validation={validation.get('status')}; engine={code.get('generation_engine')}",
                latency,
            )
        finally:
            request_json(
                args.base_url,
                "/memory/clear",
                method="POST",
                payload={"session_id": session_id},
                token=token,
                timeout=args.timeout,
            )

    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = len(results) - passed
    timings = [elapsed for _, ok, _, elapsed in results if ok and elapsed is not None]
    if timings:
        print(
            f"\nSUMMARY: {passed}/{len(results)} passed; {failed} failed; "
            f"median measured request latency={statistics.median(timings):.0f} ms"
        )
    else:
        print(f"\nSUMMARY: {passed}/{len(results)} passed; {failed} failed; no latency samples")
    if not args.live_model:
        print("Model-quality probes were skipped. Use --live-model only when provider usage is acceptable.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
