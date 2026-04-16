FROM python:3.10-bookworm

WORKDIR /app

RUN useradd -m -u 1000 user && \
    mkdir -p /app/models /app/artifacts && \
    chown -R 1000:1000 /app /home/user

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

ENV CMAKE_ARGS="-DGGML_BLAS=OFF"
ENV CMAKE_BUILD_PARALLEL_LEVEL=1

RUN python -m pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    "llama-cpp-python==0.3.20"

RUN python -m pip install --no-cache-dir --prefer-binary "numpy==1.26.4"
RUN python -m pip install --no-cache-dir --prefer-binary "onnxruntime==1.23.2"
RUN python -m pip install --no-cache-dir --prefer-binary "fastembed==0.7.4"
RUN python -m pip install --no-cache-dir --prefer-binary "gradio==5.25.0"
RUN python -m pip install --no-cache-dir --prefer-binary "llama-index-core==0.12.52.post1"
RUN python -m pip install --no-cache-dir --prefer-binary "huggingface_hub>=0.23"

COPY --chown=1000:1000 app.py .
COPY --chown=1000:1000 lab03/ ./lab03/
COPY --chown=1000:1000 artifacts/ ./artifacts/

USER 1000

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    MODEL_FILENAME=Qwen3.5-4B.Q2_K.gguf \
    LLM_CONTEXT_WINDOW=1024 \
    LLM_BATCH_SIZE=64 \
    LLM_THREADS=4

RUN python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Jackrong/Qwen3.5-4B-Neo-GGUF",
    filename="Qwen3.5-4B.Q2_K.gguf",
    local_dir="/app/models",
)
EOF

EXPOSE 7860

CMD ["python", "app.py"]