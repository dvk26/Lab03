FROM python:3.10-bookworm

WORKDIR /app

# uid 1000 is required by HF Spaces Docker runtime
RUN useradd -m -u 1000 user && \
    mkdir -p /app/models /app/artifacts && \
    chown -R 1000:1000 /app /home/user

# Use the prebuilt CPU wheel only; never compile llama.cpp from source in Spaces.
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.19" \
    --only-binary=llama-cpp-python \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

COPY app.py .
COPY lab03/ ./lab03/
COPY artifacts/ ./artifacts/

USER 1000

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

CMD ["python", "app.py"]
