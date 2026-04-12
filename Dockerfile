FROM python:3.10-slim-bookworm

WORKDIR /app

# uid 1000 is required by HF Spaces Docker runtime
RUN useradd -m -u 1000 user
RUN mkdir -p /app/models /app/artifacts && chown -R user:user /app /home/user

# libgomp1 required by ONNXRuntime/fastembed (one small package)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# llama-cpp-python: binary wheel ONLY — never compile from source.
# Compiling C++ OOMKills the HF free-tier builder (2 GB RAM limit).
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.19" \
    --only-binary=llama-cpp-python \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Install runtime deps one-by-one so the failing step is visible in build error output
RUN pip install --no-cache-dir "numpy>=1.26.0"
RUN pip install --no-cache-dir "huggingface-hub>=0.30.0"
RUN pip install --no-cache-dir "fastembed>=0.4.0"
RUN pip install --no-cache-dir "llama-index-core>=0.10.0,<0.13.0"
RUN pip install --no-cache-dir "gradio>=5.0.0"

COPY --chown=user app.py .
COPY --chown=user pyproject.toml .
COPY --chown=user lab03/ ./lab03/
COPY --chown=user artifacts/ ./artifacts/

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

CMD ["python", "app.py"]
