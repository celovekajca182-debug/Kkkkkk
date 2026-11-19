# SWIN AI: serious mathematics and coding fine-tuning

## Why this training path exists

The legacy train_transformer.py initializes a small decoder from random weights. It is useful for experiments with SWIN's tokenizer and attention implementation, but the small local corpus cannot give it broad mathematical and programming competence.

The new train_swin_qlora.py fine-tunes the pretrained Qwen/Qwen3-1.7B model using 4-bit QLoRA. The output is a PEFT adapter, not a copy of the full base weights.

## Curriculum — no greeting filler

- Advanced mathematics: project-numina/NuminaMath-CoT (Apache-2.0), with olympiad, AIME/AMC, AoPS and other problem sources; plus EleutherAI/hendrycks_math (MIT) across seven mathematical areas.
- Programming: NVIDIA OpenCodeReasoning split_0 (CC BY 4.0; train plus deterministic holdout from train or the source validation split; official test excluded), MBPP sanitized (CC BY 4.0), plus locally reviewed SWIN coding examples. OpenCodeReasoning contributes Python contest problems with reasoning responses and code solutions.
- Generic greeting/dialogue samples: zero in this fine-tuning stage. The base model already has general conversational capability; this limited training budget is reserved for mathematics and code.

Math prompts request rigorous derivations, justified steps, verification and a final answer. Coding prompts request maintainable implementations and respect for supplied constraints. User prompt tokens are masked from the training loss; assistant answer tokens are trained.

Dataset and model cards:
- https://huggingface.co/Qwen/Qwen3-1.7B
- https://huggingface.co/datasets/project-numina/NuminaMath-CoT
- https://huggingface.co/datasets/EleutherAI/hendrycks_math
- https://huggingface.co/datasets/google-research-datasets/mbpp
- https://huggingface.co/datasets/nvidia/OpenCodeReasoning

## Google Colab

1. Open notebooks/SWIN_AI_QLoRA_Colab.ipynb in Colab. The notebook keeps Colab's preinstalled CUDA-enabled PyTorch rather than replacing it.
2. Choose Runtime -> Change runtime type -> T4 GPU (or another CUDA GPU).
3. Run cells from top to bottom.
4. Create a Hugging Face access token with write access only if you want to publish the adapter. Log in through the Colab prompt; never paste the token into the notebook or commit it to Git.
5. Set HF_REPO_ID in the notebook to your own model repository, then run the training cell.

Command-line equivalent from the server directory:

```bash
pip install -r requirements-qlora.txt
python prepare_coding_data.py --max-train 700 --max-validation 160
python train_swin_qlora.py \
  --base-model Qwen/Qwen3-1.7B \
  --max-numina 5000 \
  --max-opencode-train 3000 \
  --max-opencode-validation 300 \
  --max-hendrycks-per-config 250 \
  --max-hendrycks-validation-per-config 100 \
  --max-seq-length 1536 \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --output-dir /content/drive/MyDrive/SWIN_AI/qlora-run
```

To publish, log in with huggingface_hub.login() and add:
    --push-to-hub --hub-model-id YOUR_USERNAME/swin-ai-qwen3-1.7b-math-code

## Limits and deployment

- CUDA is required. Free Colab GPU availability and session duration are not guaranteed; save the output directory to mounted Google Drive to preserve checkpoints.
- QLoRA trains adapter weights while keeping the quantized base weights frozen. It is much more practical on a Colab GPU than training a new language model from random initialization.
- A lower validation loss does not prove mathematical correctness or successful code execution. Keep held-out test tasks and run generated code only in a sandbox before considering the model reliable.
- The adapter is not yet connected to the Android API. Publishing it to Hugging Face stores the adapter; the inference service must load both the Qwen base model and this adapter. The Android app should be pointed to that inference API, not to the Hub model-page URL.
- The old from-scratch SWIN checkpoint is left untouched for rollback and comparison.
