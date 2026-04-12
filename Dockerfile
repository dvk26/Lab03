FROM python:3.10-bookworm

WORKDIR /app

# uid 1000 is required by HF Spaces Docker runtime
RUN useradd -m -u 1000 user && \
    mkdir -p /app/models /app/artifacts && \
    chown -R 1000:1000 /app /home/user

# llama-cpp-python: binary wheel only — never compile from source (OOMKills HF free-tier builder).
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.19" \
    --only-binary=llama-cpp-python \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Runtime deps inlined — no COPY requirements_hf.txt needed (file may not exist in HF Space repo).
RUN pip install --no-cache-dir \
    "numpy>=1.26.0" \
    "huggingface-hub>=0.30.0" \
    "fastembed>=0.4.0" \
    "llama-index-core>=0.10.0,<0.13.0" \
    "gradio>=5.0.0"

COPY --chown=1000:1000 app.py .
COPY --chown=1000:1000 lab03/ ./lab03/

USER 1000

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

CMD ["python", "app.py"]
