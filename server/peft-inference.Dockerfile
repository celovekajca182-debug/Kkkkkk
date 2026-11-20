# Run this container on a CUDA GPU host, not on the Render CPU API service.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface

WORKDIR /app
COPY server/inference-requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY server/peft_inference_server.py ./peft_inference_server.py

EXPOSE 8000
CMD ["uvicorn", "peft_inference_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
