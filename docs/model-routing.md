# SWIN AI: Qwen model routing

SWIN uses two independent model roles:

| Role | Default model | Responsibility |
| --- | --- | --- |
| General chat | `Qwen/Qwen3-4B-Thinking-2507` | Conversation, explanations, math and general reasoning |
| Coding | `Qwen/Qwen2.5-Coder-7B-Instruct` | Code generation, debugging, refactoring and repair |

The model IDs are configurable; no model weights are bundled into the Android APK or the lightweight FastAPI service.

The selected Hugging Face cards identify both defaults as Apache-2.0: Qwen3-4B-Thinking-2507 is a 4B reasoning model, while Qwen2.5-Coder-7B-Instruct has 7.61B parameters and a 32K native context (the card documents YaRN extension for longer contexts).

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

Each inference endpoint must implement `POST /v1/chat/completions`. vLLM and SGLang document this interface for the two selected model families. Recommended: run Qwen3 and Qwen2.5-Coder as separate GPU deployments and set a separate URL/key for each role. A shared gateway is also supported.
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

## Example: vLLM

Run each model on a GPU inference host and set `SWIN_CHAT_BASE_URL` and `SWIN_CODE_BASE_URL` separately, or expose both behind a model-routing gateway and use `SWIN_LLM_BASE_URL`. The official Qwen model cards include vLLM commands and OpenAI-compatible chat-completions examples. A single vLLM process started with one model serves that model only; two model IDs require two deployments or a compatible gateway.

After deployment, set the SWIN variables above in the FastAPI host's environment. Restart/redeploy the API and inspect `GET /capabilities`: the `reasoning` section reports whether the provider is configured and which model IDs are selected.

## Safety and limits

- The API key is read only by the server and is never returned by `/capabilities`.
- Generated code still goes through SWIN's existing static checks. The API does not execute user-generated programs.
- A successful API response confirms generation, not mathematical correctness or runtime correctness.
- If no GPU inference endpoint/key has been configured, changing the model IDs alone does not download or run model weights.
- Keep account memory and web/retrieval context in SWIN; the provider receives only the prompt assembled for that request.
