FROM python:3.10-slim

WORKDIR /app

# git is needed by huggingface_hub for LFS; no build-essential needed (prebuilt wheels only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# llama-cpp-python prebuilt CPU wheel (no compilation, fast)
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.20" \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Runtime deps — fastembed replaces sentence-transformers+torch (ONNX, no torch needed)
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

COPY . .

# Pure-Python package — set PYTHONPATH instead of running pip install -e .
# This avoids pip build-isolation downloading setuptools at build time.
ENV PYTHONPATH=/app

CMD ["python", "app.py"]
