"""OpenAI-compatible inference routing for the two SWIN model roles.

The API server stays lightweight; model weights live on a dedicated inference
host (vLLM, SGLang, or another OpenAI-compatible provider).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Literal

CHAT_MODEL = os.getenv(
    "SWIN_CHAT_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
).strip()
CODE_MODEL = os.getenv(
    "SWIN_CODE_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct"
).strip()
BASE_URL = os.getenv("SWIN_LLM_BASE_URL", "").strip().rstrip("/")
API_KEY = os.getenv("SWIN_LLM_API_KEY", "").strip()
TIMEOUT_SECONDS = max(
    5.0, min(240.0, float(os.getenv("SWIN_LLM_TIMEOUT_SECONDS", "90")))
)


def provider_configured() -> bool:
    """Return whether a remote OpenAI-compatible inference endpoint is configured."""
    return bool(BASE_URL)


def model_for_role(role: Literal["chat", "code"]) -> str:
    if role == "chat":
        return CHAT_MODEL
    if role == "code":
        return CODE_MODEL
    raise ValueError("Unknown SWIN model role.")


def _completion_url(base_url: str) -> str:
    """Accept a server root, an /v1 base URL, or a full chat-completions URL."""
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def generate_model_text(
    prompt: str,
    *,
    role: Literal["chat", "code"] = "chat",
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> str:
    """Generate text using the configured model role and OpenAI chat-completions API."""
    if not BASE_URL:
        raise RuntimeError("SWIN_LLM_BASE_URL is not configured.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    model = model_for_role(role)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(1, min(int(max_tokens), 8192)),
        "temperature": max(0.0, min(float(temperature), 2.0)),
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY

    request = urllib.request.Request(
        _completion_url(BASE_URL),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw_body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(600).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"SWIN inference provider returned HTTP {error.code}: {detail[:500]}"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"SWIN inference provider is unavailable: {error}") from error

    try:
        result = json.loads(raw_body.decode("utf-8"))
        message = result["choices"][0]["message"]
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Provider returned an empty assistant message.")
        return content.strip()
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise RuntimeError("SWIN inference provider returned an invalid response.") from error
    except ValueError as error:
        raise RuntimeError(str(error)) from error
