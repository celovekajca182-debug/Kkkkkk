"""Optional OpenAI Responses API provider for stronger text and code generation.

SWIN keeps its own retrieval, memory, safeguards and local model. This provider is
only used when the server owner explicitly configures SWIN_LLM_API_KEY.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class LLMProviderError(RuntimeError):
    """Safe provider failure without exposing credentials."""


def llm_enabled() -> bool:
    return bool(os.getenv("SWIN_LLM_API_KEY", "").strip())


def _extract_output_text(response: dict) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for block in item.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    return "\n".join(parts).strip()


def generate_text(
    prompt: str,
    *,
    instructions: str,
    max_output_tokens: int = 1200,
    model_override: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Generate text through the Responses API without storing provider state."""
    api_key = os.getenv("SWIN_LLM_API_KEY", "").strip()
    if not api_key:
        raise LLMProviderError("SWIN_LLM_API_KEY is not configured.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise LLMProviderError("The generation prompt is empty.")
    if len(prompt) > 60_000:
        raise LLMProviderError("The generation prompt exceeds the configured limit.")

    model = (model_override or os.getenv("SWIN_REASONING_MODEL", "gpt-6.2-sol")).strip()
    if not model or len(model) > 120:
        raise LLMProviderError("Invalid reasoning model identifier.")
    output_limit = max(64, min(int(max_output_tokens), 16384))
    effort = (reasoning_effort or os.getenv("SWIN_REASONING_EFFORT", "medium")).strip().lower()
    if effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise LLMProviderError("Invalid reasoning effort setting.")
    payload = {
        "model": model,
        "input": [
            {"role": "developer", "content": instructions[:6000]},
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": output_limit,
        "reasoning": {"effort": effort},
        "store": False,
    }
    request = urllib.request.Request(
        os.getenv("SWIN_LLM_ENDPOINT", "https://api.openai.com/v1/responses"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SWIN-AI-Reasoning/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read(3 * 1024 * 1024)
        result = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            details = error.read(1000).decode("utf-8", errors="replace")
        except Exception:
            details = ""
        print("SWIN LLM provider HTTP error:", error.code, details[:400])
        if error.code in {401, 403}:
            raise LLMProviderError("Провайдер SWIN отклонил API-ключ. Проверь SWIN_LLM_API_KEY.") from error
        if error.code == 429:
            raise LLMProviderError("Лимит запросов или средств провайдера SWIN исчерпан.") from error
        raise LLMProviderError("Провайдер SWIN временно вернул ошибку.") from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        print("SWIN LLM provider connection error:", str(error)[:300])
        raise LLMProviderError("Не удалось связаться с провайдером SWIN.") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LLMProviderError("Провайдер SWIN вернул некорректный ответ.") from error

    output = _extract_output_text(result)
    if not output:
        raise LLMProviderError("Провайдер SWIN не вернул текстовый ответ.")
    return output[:100_000]
