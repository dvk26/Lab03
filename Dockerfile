FROM python:3.10-slim

WORKDIR /app

# git is needed by huggingface_hub for LFS; libgomp1 is needed by ONNXRuntime/fastembed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# llama-cpp-python prebuilt CPU wheel (no compilation, fast)
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.19" \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Runtime deps — fastembed replaces sentence-transformers+torch (ONNX, no torch needed)
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy only runtime-needed files — avoids pulling in storage/, models/, evaluation/, etc.
COPY app.py .
COPY pyproject.toml .
COPY lab03/ ./lab03/
COPY artifacts/ ./artifacts/

ENV PYTHONPATH=/app

CMD ["python", "app.py"]
