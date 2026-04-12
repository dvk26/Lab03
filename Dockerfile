FROM python:3.10-slim-bookworm

WORKDIR /app

# uid 1000 is required by HF Spaces Docker runtime
RUN useradd -m -u 1000 user
RUN mkdir -p /app/models /app/artifacts && chown -R user:user /app /home/user

# libgomp1 required by ONNXRuntime/fastembed (one small package)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# llama-cpp-python: binary wheel ONLY — never compile from source.
# Compiling C++ OOMKills the HF free-tier builder (2 GB RAM limit).
# If no prebuilt wheel matches, pip will exit with a clean error instead of OOMKilling.
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.19" \
    --only-binary=llama-cpp-python \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Runtime deps
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

COPY --chown=user app.py .
COPY --chown=user pyproject.toml .
COPY --chown=user lab03/ ./lab03/
COPY --chown=user artifacts/ ./artifacts/

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

CMD ["python", "app.py"]
