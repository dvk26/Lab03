FROM python:3.10-bookworm

WORKDIR /app

# uid 1000 is required by HF Spaces Docker runtime
RUN useradd -m -u 1000 user && \
    mkdir -p /app/models /app/artifacts && \
    chown -R 1000:1000 /app /home/user

# llama-cpp-python: binary wheel ONLY — never compile from source (OOMKills HF free-tier builder).
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.19" \
    --only-binary=llama-cpp-python \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Runtime deps inlined — avoids COPY requirements_hf.txt failures on HF BuildKit.
# onnxruntime is NOT pinned here — fastembed picks a compatible version automatically.
RUN pip install --no-cache-dir \
    "fastembed==0.7.4" \
    "gradio==5.25.0" \
    "huggingface-hub==0.30.2" \
    "llama-index-core==0.12.52.post1" \
    "numpy==1.26.4"

COPY --chown=1000:1000 app.py .
COPY --chown=1000:1000 lab03/ ./lab03/

USER 1000

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

CMD ["python", "app.py"]
