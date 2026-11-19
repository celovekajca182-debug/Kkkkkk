# SWIN AI: Qwen model routing

SWIN uses two independent model roles:

| Role | Default model | Responsibility |
| --- | --- | --- |
| General chat | `Qwen/Qwen3-4B-Instruct-2507` | Conversation, explanations, math and general reasoning |
| Coding | `Qwen/Qwen2.5-Coder-3B-Instruct` | Code generation, debugging, refactoring and repair |

The model IDs are configurable; no model weights are bundled into the Android APK or the lightweight FastAPI service.

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

The inference endpoint must implement `POST /v1/chat/completions`. vLLM and SGLang document this interface for the two selected model families. Configure one compatible endpoint that can route both model IDs, or use a gateway that exposes both deployments.

## Environment variables

Set these as server-side secrets/environment variables. Never put the provider key in the Android app or commit it to Git.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SWIN_LLM_BASE_URL` | empty | OpenAI-compatible API root, `/v1` URL, or full `/chat/completions` URL |
| `SWIN_LLM_API_KEY` | empty | Optional bearer key for the inference provider |
| `SWIN_CHAT_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Model ID used by `/chat` |
| `SWIN_CODE_MODEL` | `Qwen/Qwen2.5-Coder-3B-Instruct` | Model ID used by `/code/assist` |
| `SWIN_LLM_TIMEOUT_SECONDS` | `90` | Inference request timeout, clamped to 5–240 seconds |

When `SWIN_LLM_BASE_URL` is empty, SWIN preserves the legacy local-checkpoint path. When it is set, Qwen is preferred; if the provider is unavailable, the existing local model/retrieval fallback remains in place.

## Example: vLLM

Run each model on a GPU inference host (or expose them behind a model-routing gateway). The official Qwen model cards include vLLM commands and OpenAI-compatible chat-completions examples. A single vLLM process started with one model serves that model only; two model IDs require two deployments or a compatible gateway.

After deployment, set the SWIN variables above in the FastAPI host's environment. Restart/redeploy the API and inspect `GET /capabilities`: the `reasoning` section reports whether the provider is configured and which model IDs are selected.

## Safety and limits

- The API key is read only by the server and is never returned by `/capabilities`.
- Generated code still goes through SWIN's existing static checks. The API does not execute user-generated programs.
- A successful API response confirms generation, not mathematical correctness or runtime correctness.
- If no GPU inference endpoint/key has been configured, changing the model IDs alone does not download or run model weights.
- Keep account memory and web/retrieval context in SWIN; the provider receives only the prompt assembled for that request.
