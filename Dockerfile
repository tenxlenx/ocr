# CUDA 13.0.1 runtime (Ubuntu 22.04) + cu130 wheels
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/data/hf_cache

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch from cu130 channel
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu130 \
        torch torchvision

# Core libs for the OCR model + API (+ python-multipart fix + python-dotenv)
RUN python3 -m pip install \
      transformers==4.46.3 \
      tokenizers==0.20.3 \
      fastapi "uvicorn[standard]" \
      python-multipart \
      python-dotenv \
      accelerate bitsandbytes safetensors pillow \
      huggingface_hub einops addict easydict

# Optional: flash-attn (ignore failure if no wheels/toolchain)
RUN python3 - <<'PY' || true
try:
    import subprocess
    subprocess.check_call(["python3","-m","pip","install","--no-build-isolation","flash-attn==2.7.3"])
except Exception as e:
    print("flash-attn install skipped:", e)
PY

# App
COPY ocr_api_allinone.py /app/ocr_api_allinone.py

# Create default data dirs
RUN mkdir -p /app/results /app/uploads /data/hf_cache

# Expose default; actual port is controlled by PORT in .env + compose
EXPOSE 8000

# Run the server (reads .env inside the script)
CMD ["python3", "-u", "ocr_api_allinone.py"]

