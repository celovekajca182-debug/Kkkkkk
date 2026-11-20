# SWIN-AI-Coder-Smart: GPU inference deployment

The SWIN API on Render remains the lightweight gateway. The custom LoRA model
must run on a separate CUDA GPU service: the adapter is not a standalone model
and the Render CPU API must not try to load its 4-bit weights.

## Exact model pairing

- Base model: unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit
- LoRA adapter: MAKSIM67771488/SWIN-AI-Coder-Smart
- Adapter type: LoRA, rank 32, alpha 32
- Architecture: Qwen2ForCausalLM
- Runtime: Transformers + PEFT + bitsandbytes on CUDA

The server loads the base model once, then applies the adapter with
PeftModel.from_pretrained. It uses the tokenizer/chat template stored in the
adapter repository. The model weights are downloaded from Hugging Face at
container startup and cached under /data/huggingface.

## Build and run the GPU inference service

Build from the repository root:

    docker build -f server/peft-inference.Dockerfile -t swin-peft-coder .

Run on a CUDA-enabled host with access to an NVIDIA GPU:

    docker run --gpus all -p 8000:8000 \
      -e SWIN_INFERENCE_API_KEY=YOUR_LONG_RANDOM_SERVER_KEY \
      -e HF_TOKEN=YOUR_HF_READ_TOKEN \
      -v swin-hf-cache:/data/huggingface \
      swin-peft-coder

HF_TOKEN is only required if Hugging Face requires authentication for the model
repositories. Never commit either token to Git or put it in the Android app.
Use one inference process/worker per GPU replica; the service serializes
generation requests to avoid concurrent GPU memory spikes.

The container exposes:
- GET /health (readiness and model identity; no credentials)
- GET /v1/models
- POST /v1/chat/completions (OpenAI-compatible, non-streaming)

## Connect the SWIN API on Render

Set these server-side environment variables on the existing SWIN API service:

    SWIN_CODE_BASE_URL=https://YOUR-GPU-INFERENCE-HOST/v1
    SWIN_CODE_API_KEY=THE_SAME_LONG_RANDOM_SERVER_KEY
    SWIN_CODE_MODEL=MAKSIM67771488/SWIN-AI-Coder-Smart

Keep the chat route configured separately. For example, SWIN_CHAT_BASE_URL
may continue to use the Hugging Face router for a supported chat model.
The code-specific URL overrides SWIN_LLM_BASE_URL for code requests.

After the GPU service is ready and these values are saved on Render, redeploy
the SWIN API. Check GET /capabilities: code_provider_enabled must be true and
coding_model must show MAKSIM67771488/SWIN-AI-Coder-Smart.

## Important operational notes

- The Hugging Face Inference Providers router serves models available through
  its provider catalog; it does not automatically load an arbitrary PEFT
  adapter from a repository. This custom adapter needs a runtime that loads
  both the base and adapter, which is what this container does.
- A public GPU host can incur charges while running. Choose a small CUDA GPU
  with enough VRAM for the 4-bit base plus runtime overhead. Confirm the host's
  current GPU/driver support before deploying.
- This implementation has not yet been run against a live GPU in this repo.
  A healthy container startup and a real prompt test are required before
  claiming the adapter is active.


## Deploy this container on Hugging Face Inference Endpoints

The Docker image is published to GitHub Container Registry by
`.github/workflows/publish-swin-peft-image.yml`.
Use the immutable commit tag shown in the successful publish workflow run:
`ghcr.io/celovekajca182-debug/swin-peft-coder:c63a4fce3448623e2e03514adea0595d0caeade1`.

1. Open the [Inference Endpoints dashboard](https://ui.endpoints.huggingface.co/).
2. Select the base repository `unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit`.
   Hugging Face mounts this repository at `/repository`; the SWIN server
   detects that mount and loads the base model locally.
3. Choose GPU hardware `AWS / NVIDIA T4 / x1` (14 GB, listed at $0.50/hour).
   Select the nearest EU region offered in the region picker.
4. Under **Custom Container**, use the published image tag, set port `8000`,
   and health route `/health`.
5. Add runtime environment variables:
   - `SWIN_INFERENCE_API_KEY`: use the same value as the existing Render
     `SWIN_LLM_API_KEY`, or create a separate long random key and then also set
     `SWIN_CODE_API_KEY` to that same value on Render.
   - `HF_TOKEN`: optional read token; the base and adapter repositories are
     public, so it is not needed unless Hub access requires authentication.
   - `SWIN_CODE_ADAPTER=MAKSIM67771488/SWIN-AI-Coder-Smart`
   - `SWIN_BASE_MODEL=/repository`
6. Set one maximum replica. Enable **Automatic Scale-to-Zero** and minimum
   replicas `0` to avoid paying while idle. A scale-from-zero request has a
   cold start; the API sends `X-Scale-Up-Timeout` to give the model time to load.
7. Create the endpoint. Copy its HTTPS endpoint URL when the status is Running.

Hugging Face requires an active subscription and payment method for dedicated
Inference Endpoints. T4 is billed at $0.50/hour while the replica is
initializing or running; billing is per minute. Scale-to-zero stops compute
charges while idle, but the first request after idle can wait for model startup.

After the endpoint URL exists, set these server-side environment variables on
the existing Render API service (never in the Android APK):

    SWIN_CODE_BASE_URL=https://YOUR-ENDPOINT-URL/v1
    # Set only when you chose a new key instead of reusing SWIN_LLM_API_KEY:
    SWIN_CODE_API_KEY=THE_SAME_SWIN_INFERENCE_API_KEY
    SWIN_CODE_MODEL=MAKSIM67771488/SWIN-AI-Coder-Smart

Then redeploy the Render API and verify `/capabilities` reports
`code_provider_enabled: true` and the SWIN adapter ID as `coding_model`.
Do not set the GPU endpoint to always-on unless you explicitly accept the
continuous hourly charge.
