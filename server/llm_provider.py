"""OpenAI-compatible inference routing for the two SWIN model roles.

The API server stays lightweight; model weights live on dedicated inference
hosts (vLLM, SGLang, or another OpenAI-compatible provider).
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
    "SWIN_CODE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
).strip()

# A shared gateway is supported, but per-role URLs allow two independent GPU hosts.
BASE_URL = os.getenv("SWIN_LLM_BASE_URL", "").strip().rstrip("/")
CHAT_BASE_URL = os.getenv("SWIN_CHAT_BASE_URL", "").strip().rstrip("/")
CODE_BASE_URL = os.getenv("SWIN_CODE_BASE_URL", "").strip().rstrip("/")
API_KEY = os.getenv("SWIN_LLM_API_KEY", "").strip()
CHAT_API_KEY = os.getenv("SWIN_CHAT_API_KEY", "").strip()
CODE_API_KEY = os.getenv("SWIN_CODE_API_KEY", "").strip()
try:
    TIMEOUT_SECONDS = max(
        5.0, min(240.0, float(os.getenv("SWIN_LLM_TIMEOUT_SECONDS", "90")))
    )
except ValueError:
    TIMEOUT_SECONDS = 90.0


def _base_url_for_role(role: Literal["chat", "code"]) -> str:
    if role == "chat":
        return CHAT_BASE_URL or BASE_URL
    if role == "code":
        return CODE_BASE_URL or BASE_URL
    raise ValueError("Unknown SWIN model role.")


def _api_key_for_role(role: Literal["chat", "code"]) -> str:
    if role == "chat":
        return CHAT_API_KEY or API_KEY
    if role == "code":
        return CODE_API_KEY or API_KEY
    raise ValueError("Unknown SWIN model role.")


def provider_configured(role: Literal["chat", "code"] | None = None) -> bool:
    """Return whether the shared endpoint or the selected role endpoint is set."""
    if role is None:
        return bool(_base_url_for_role("chat") or _base_url_for_role("code"))
    return bool(_base_url_for_role(role))


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
    """Generate text using the selected role's OpenAI-compatible chat-completions API."""
    base_url = _base_url_for_role(role)
    if not base_url:
        raise RuntimeError(f"SWIN {role} inference endpoint is not configured.")
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
    api_key = _api_key_for_role(role)
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    request = urllib.request.Request(
        _completion_url(base_url),
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
