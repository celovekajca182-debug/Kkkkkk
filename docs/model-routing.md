# SWIN AI: model routing

SWIN uses two independent model roles:

| Role | Default model | Responsibility |
| --- | --- | --- |
| General chat | `Qwen/Qwen3-4B-Thinking-2507` | Conversation, explanations, math and general reasoning |
| Coding | `MAKSIM67771488/SWIN-AI-Coder-Smart` | Custom SWIN LoRA coder loaded on its matching Qwen2.5-Coder-1.5B base |

The model IDs are configurable; no model weights are bundled into the Android APK or the lightweight FastAPI service.

The chat route uses a provider-served Qwen3 model. The coding route defaults to the SWIN LoRA adapter; it must be loaded together with its exact 4-bit base model by the dedicated CUDA service described in `docs/swin-adapter-deployment.md`.

## Runtime architecture

```text
Android app
    |
    v
SWIN FastAPI (Render / CPU)
    |-- /chat ------> Qwen3 chat model
    |-- /code/assist -> SWIN PEFT inference service (CUDA GPU)
    |-- memory, auth, retrieval and static code checks stay in SWIN
    |
    v
OpenAI-compatible inference endpoint (GPU/provider)
```

Each inference endpoint must implement `POST /v1/chat/completions`.
Example vLLM commands (run on separate GPU hosts or expose through a router):

```bash
# Chat / deep reasoning endpoint
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
  --host 0.0.0.0 --port 8000 \
  --enable-reasoning --reasoning-parser deepseek_r1

# Coding endpoint (separate host)
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --host 0.0.0.0 --port 8000
```

Set `SWIN_CHAT_BASE_URL=https://CHAT-HOST/v1` and `SWIN_CODE_BASE_URL=https://CODE-HOST/v1` on the FastAPI service. The API key variables are optional if your inference hosts do not require bearer authentication.
 vLLM and SGLang document this interface for the two selected model families. Recommended: run Qwen3 and Qwen2.5-Coder as separate GPU deployments and set a separate URL/key for each role. A shared gateway is also supported.
For the Qwen3 Thinking endpoint, enable the reasoning parser (vLLM: `--enable-reasoning --reasoning-parser deepseek_r1`; SGLang: `--reasoning-parser deepseek-r1`) so internal reasoning is separated from the user-visible answer.

## Environment variables

Set these as server-side secrets/environment variables. Never put the provider key in the Android app or commit it to Git.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SWIN_LLM_BASE_URL` | empty | OpenAI-compatible API root, `/v1` URL, or full `/chat/completions` URL |
| `SWIN_LLM_API_KEY` | empty | Shared bearer key for a gateway |
| `SWIN_CHAT_BASE_URL` | shared URL | Dedicated Qwen3 endpoint; overrides the shared URL |
| `SWIN_CODE_BASE_URL` | shared URL | Dedicated Qwen2.5-Coder endpoint; overrides the shared URL |
| `SWIN_CHAT_API_KEY` | shared key | Optional Qwen3 endpoint key |
| `SWIN_CODE_API_KEY` | shared key | Optional Qwen2.5-Coder endpoint key |
| `SWIN_CHAT_MODEL` | `Qwen/Qwen3-4B-Thinking-2507` | Model ID used by `/chat` |
| `SWIN_CODE_MODEL` | `MAKSIM67771488/SWIN-AI-Coder-Smart` | Custom adapter served by the dedicated CUDA service |
| `SWIN_LLM_TIMEOUT_SECONDS` | `90` | Inference request timeout, clamped to 5–240 seconds |
| `SWIN_CHAT_MAX_TOKENS` | `4096` | Maximum answer tokens for Qwen3 |
| `SWIN_CODE_MAX_TOKENS` | `2400` | Maximum generated code tokens |
| `SWIN_CHAT_TEMPERATURE` | `0.25` | Sampling temperature for chat |
| `SWIN_CODE_TEMPERATURE` | `0.15` | Sampling temperature for code |

When a role-specific URL and the shared `SWIN_LLM_BASE_URL` are all empty, SWIN preserves the legacy local-checkpoint path for that role. A role-specific URL overrides the shared gateway URL. If the configured provider is unavailable, the existing local model/retrieval fallback remains in place.

## Easiest setup: Hugging Face Inference Providers

Hugging Face's Inference Providers router chooses a provider for catalog models. SWIN can use it for general chat. It does not automatically load this custom PEFT adapter from its repository; the SWIN coding role therefore requires the dedicated CUDA service and role-specific URL documented in `docs/swin-adapter-deployment.md`. Do not send the adapter ID to the shared HF router.

In the Render API service environment, set:

```text
SWIN_LLM_BASE_URL=https://router.huggingface.co/v1
SWIN_LLM_API_KEY=hf_your_inference_token
SWIN_CHAT_MODEL=Qwen/Qwen3-4B-Thinking-2507
SWIN_CODE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
```

Create a Hugging Face token with inference permissions, enable an available provider in the Inference Providers settings, add these values to the server (not the Android app), and redeploy the API. This shared router configuration serves chat only while the custom adapter is the configured code model. Set `SWIN_CODE_BASE_URL` and `SWIN_CODE_API_KEY` separately after deploying the dedicated CUDA service. Provider usage is billed at the selected provider's price; check current pricing before using it.

## Example: vLLM

Run the general chat model on a provider/GPU host as needed. The SWIN adapter requires the dedicated PEFT container described in `docs/swin-adapter-deployment.md`; set `SWIN_CODE_BASE_URL` explicitly. Do not use one generic Hugging Face router URL for both roles.

After deployment, set the SWIN variables above in the FastAPI host's environment. Restart/redeploy the API and inspect `GET /capabilities`: the `reasoning` section reports whether the provider is configured and which model IDs are selected.

## Safety and limits

- The API key is read only by the server and is never returned by `/capabilities`. Account limits are 12 chat requests, 5 code generations, 8 scaffold generations, 8 memory synchronizations and 20 memory fact writes/deletes per five-minute window. The in-process limiter is capped at 4,096 active keys and is not a distributed quota system. Memory restore requests are capped at 250,000 aggregate characters.
- Generated code still goes through SWIN's existing static checks. The API does not execute user-generated programs.
- A successful API response confirms generation, not mathematical correctness or runtime correctness.
- If no GPU inference endpoint/key has been configured, changing the model IDs alone does not download or run model weights.
- Keep account memory and web/retrieval context in SWIN; the provider receives only the prompt assembled for that request.
