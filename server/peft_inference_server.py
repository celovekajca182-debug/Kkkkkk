"""OpenAI-compatible GPU inference server for the SWIN PEFT adapter.

Run this service on a CUDA host. Keep the lightweight SWIN API on Render/CPU and
point SWIN_CODE_BASE_URL at this service. The 4-bit base model is loaded once
at startup and the SWIN LoRA adapter is applied with PEFT.
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

LOG = logging.getLogger("swin.peft.inference")

BASE_MODEL_ID = "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
ADAPTER_ID = "MAKSIM67771488/SWIN-AI-Coder-Smart"
MODEL_ID = ADAPTER_ID
INFERENCE_API_KEY = ""
import os
BASE_MODEL_ID = os.getenv("SWIN_BASE_MODEL", BASE_MODEL_ID).strip()
ADAPTER_ID = os.getenv("SWIN_CODE_ADAPTER", ADAPTER_ID).strip()
MODEL_ID = os.getenv("SWIN_CODE_MODEL", ADAPTER_ID).strip()
INFERENCE_API_KEY = os.getenv("SWIN_INFERENCE_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", "")).strip()

MAX_CONTEXT_TOKENS = max(2048, min(32768, int(os.getenv("SWIN_MAX_CONTEXT_TOKENS", "32768"))))
DEFAULT_MAX_NEW_TOKENS = max(1, min(8192, int(os.getenv("SWIN_CODE_MAX_TOKENS", "2400"))))

app = FastAPI(title="SWIN PEFT Coder", version="1.0.0")
_runtime: tuple[Any, Any, Any] | None = None
_generation_lock = threading.Lock()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=8192)
    temperature: float = Field(default=0.15, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    stream: bool = False


def _authorize(authorization: str | None) -> None:
    """Require a server-to-server key; never expose it through an endpoint."""
    if not INFERENCE_API_KEY:
        raise HTTPException(status_code=503, detail="Inference API key is not configured.")
    expected = "Bearer " + INFERENCE_API_KEY
    if not authorization or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid inference credentials.")


def _check_model(requested_model: str | None) -> str:
    if requested_model and requested_model not in {MODEL_ID, ADAPTER_ID, "swin-ai-coder"}:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint serves only {MODEL_ID}.",
        )
    return MODEL_ID


def _apply_chat_template(tokenizer: Any, messages: list[ChatMessage]) -> str:
    """Use the adapter repository's tokenizer/template, with a Qwen fallback."""
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
    return tokenizer.apply_chat_template(
        [message.model_dump() for message in messages],
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_runtime() -> tuple[Any, Any, Any]:
    """Load the Qwen base and SWIN LoRA adapter once on a CUDA device."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The SWIN PEFT coder requires a CUDA GPU. Do not run this 4-bit model on the Render CPU service."
        )

    LOG.info("Loading SWIN tokenizer from %s", ADAPTER_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_ID,
        token=HF_TOKEN or None,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOG.info("Loading 4-bit base model %s", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN or None,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    LOG.info("Applying SWIN LoRA adapter %s", ADAPTER_ID)
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_ID,
        token=HF_TOKEN or None,
        is_trainable=False,
    )
    model.eval()
    return model, tokenizer, torch


@app.on_event("startup")
def startup_load_model() -> None:
    global _runtime
    if not INFERENCE_API_KEY:
        raise RuntimeError(
            "Set SWIN_INFERENCE_API_KEY before exposing the inference server."
        )
    _runtime = _load_runtime()
    LOG.info("SWIN PEFT coder is ready: %s", MODEL_ID)


@app.get("/health")
def health() -> dict[str, Any]:
    ready = _runtime is not None
    return {
        "status": "ok" if ready else "starting",
        "ready": ready,
        "model": MODEL_ID,
        "base_model": BASE_MODEL_ID,
        "adapter": ADAPTER_ID,
        "device": "cuda" if ready else None,
    }


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _authorize(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "swin-ai",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _authorize(authorization)
    served_model = _check_model(request.model)
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not enabled on this SWIN inference endpoint yet.",
        )
    if _runtime is None:
        raise HTTPException(status_code=503, detail="SWIN coder model is not ready.")

    model, tokenizer, torch = _runtime
    prompt = _apply_chat_template(tokenizer, request.messages)

    with _generation_lock:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        input_token_count = int(input_ids.shape[-1])
        model_context = getattr(model.config, "max_position_embeddings", MAX_CONTEXT_TOKENS)
        context_limit = min(MAX_CONTEXT_TOKENS, int(model_context))
        if input_token_count + request.max_tokens > context_limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Prompt plus requested output exceeds the {context_limit}-token context limit. "
                    "Shorten the prompt or reduce max_tokens."
                ),
            )

        input_device = model.get_input_embeddings().weight.device
        input_ids = input_ids.to(input_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_device)

        generation_options: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if request.temperature > 0:
            generation_options["temperature"] = request.temperature
            generation_options["top_p"] = request.top_p

        with torch.inference_mode():
            generated = model.generate(**generation_options)

        new_tokens = generated[0, input_token_count:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = int(new_tokens.shape[-1])
        finish_reason = "length" if completion_tokens >= request.max_tokens else "stop"

    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": served_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": input_token_count,
            "completion_tokens": completion_tokens,
            "total_tokens": input_token_count + completion_tokens,
        },
    }
