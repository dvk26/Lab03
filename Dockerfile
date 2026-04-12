FROM python:3.10-bookworm

WORKDIR /app

RUN useradd -m -u 1000 user && \
    mkdir -p /app/models /app/artifacts && \
    chown -R 1000:1000 /app /home/user

# musl: abetlen's prebuilt CPU wheels link against musl libc; install it so
# libllama.so can find libc.musl-x86_64.so.1 on this glibc (Debian) host
RUN apt-get update && \
    apt-get install -y --no-install-recommends musl && \
    rm -rf /var/lib/apt/lists/*

# llama-cpp-python: use --extra-index-url (not --find-links) for abetlen's index page
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.19" \
    --only-binary=llama-cpp-python \
    --prefer-binary \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

RUN pip install --no-cache-dir \
    "fastembed==0.7.4" \
    "gradio==5.25.0" \
    "huggingface-hub==0.30.2" \
    "llama-index-core==0.12.52.post1" \
    "numpy==1.26.4"

COPY --chown=1000:1000 app.py .
COPY --chown=1000:1000 lab03/ ./lab03/
COPY --chown=1000:1000 artifacts/ ./artifacts/

USER 1000

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
