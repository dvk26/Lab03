FROM python:3.10-bookworm

WORKDIR /app

RUN useradd -m -u 1000 user && \
    mkdir -p /app/models /app/artifacts && \
    chown -R 1000:1000 /app /home/user

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.20" \
    --only-binary=llama-cpp-python \
    --prefer-binary \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# split to isolate the failing package
RUN pip install --no-cache-dir -v "numpy==1.26.4"
RUN pip install --no-cache-dir -v "onnxruntime==1.23.2"
RUN pip install --no-cache-dir -v "huggingface-hub==0.30.2"
RUN pip install --no-cache-dir -v "fastembed==0.7.4"
RUN pip install --no-cache-dir -v "gradio==5.25.0"
RUN pip install --no-cache-dir -v "llama-index-core==0.12.52.post1"

RUN pip check

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
