# SWIN AI: Qwen model routing

SWIN uses two independent model roles:

| Role | Default model | Responsibility |
| --- | --- | --- |
| General chat | `Qwen/Qwen3-4B-Thinking-2507` | Conversation, explanations, math and general reasoning |
| Coding | `Qwen/Qwen2.5-Coder-7B-Instruct` | Code generation, debugging, refactoring and repair |

The model IDs are configurable; no model weights are bundled into the Android APK or the lightweight FastAPI service.

The selected Hugging Face model cards identify both defaults as Apache-2.0: Qwen3-4B-Thinking-2507 has 4.0B parameters and a native 262,144-token context; Qwen2.5-Coder-7B-Instruct has 7.61B parameters and a 131,072-token context.

## Runtime architecture

```text
Android app
    |
    v
SWIN FastAPI (Render / CPU)
    |-- /chat ------> Qwen3 chat model
    |-- /code/assist -> Qwen2.5-Coder model
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
| `SWIN_CODE_MODEL` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Model ID used by `/code/assist` |
| `SWIN_LLM_TIMEOUT_SECONDS` | `90` | Inference request timeout, clamped to 5–240 seconds |
| `SWIN_CHAT_MAX_TOKENS` | `4096` | Maximum answer tokens for Qwen3 |
| `SWIN_CODE_MAX_TOKENS` | `2400` | Maximum generated code tokens |
| `SWIN_CHAT_TEMPERATURE` | `0.25` | Sampling temperature for chat |
| `SWIN_CODE_TEMPERATURE` | `0.15` | Sampling temperature for code |

When a role-specific URL and the shared `SWIN_LLM_BASE_URL` are all empty, SWIN preserves the legacy local-checkpoint path for that role. A role-specific URL overrides the shared gateway URL. If the configured provider is unavailable, the existing local model/retrieval fallback remains in place.

## Easiest setup: Hugging Face Inference Providers

The current Hugging Face model cards list Featherless AI as an Inference Provider for both selected models: Qwen3-4B-Thinking-2507 and Qwen2.5-Coder-7B-Instruct. Hugging Face exposes an OpenAI-compatible router, so SWIN can use one URL and one token while still sending the selected model ID for each role. Provider availability and price may change; re-check each model card and the account's remaining Inference Providers credits before enabling live tests.

In the Render API service environment, set:

```text
SWIN_LLM_BASE_URL=https://router.huggingface.co/v1
SWIN_LLM_API_KEY=hf_your_inference_token
SWIN_CHAT_MODEL=Qwen/Qwen3-4B-Thinking-2507
SWIN_CODE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
```

Create a Hugging Face token with inference permissions, enable an available provider in the Inference Providers settings, add these values to the server (not the Android app), and redeploy the API. Hugging Face routes chat-completion requests to a provider serving the model. Provider usage is billed at that provider's price; check current pricing before using it. You do not need to configure `SWIN_CHAT_BASE_URL` or `SWIN_CODE_BASE_URL` when using the shared router.

## Example: vLLM

Run each model on a GPU inference host and set `SWIN_CHAT_BASE_URL` and `SWIN_CODE_BASE_URL` separately, or expose both behind a model-routing gateway and use `SWIN_LLM_BASE_URL`. The official Qwen model cards include vLLM commands and OpenAI-compatible chat-completions examples. A single vLLM process started with one model serves that model only; two model IDs require two deployments or a compatible gateway.

After deployment, set the SWIN variables above in the FastAPI host's environment. Restart/redeploy the API and inspect `GET /capabilities`: the `reasoning` section reports whether the provider is configured and which model IDs are selected.

## Safety and limits

- The API key is read only by the server and is never returned by `/capabilities`.
- Generated code still goes through SWIN's existing static checks. The API does not execute user-generated programs.
- A successful API response confirms generation, not mathematical correctness or runtime correctness.
- If no GPU inference endpoint/key has been configured, changing the model IDs alone does not download or run model weights.
- Keep account memory and web/retrieval context in SWIN; the provider receives only the prompt assembled for that request.
