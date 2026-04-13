FROM python:3.10-bookworm

WORKDIR /app

RUN useradd -m -u 1000 user && \
    mkdir -p /app/models /app/artifacts && \
    chown -R 1000:1000 /app /home/user

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

RUN python -m pip install --no-cache-dir -v "llama-cpp-python==0.3.20"
RUN python -m pip install --no-cache-dir -v "numpy==1.26.4"
RUN python -m pip install --no-cache-dir -v "onnxruntime==1.23.2"
RUN python -m pip install --no-cache-dir -v "fastembed==0.7.4"
RUN python -m pip install --no-cache-dir -v "gradio==5.25.0"
RUN python -m pip install --no-cache-dir -v "llama-index-core==0.12.52.post1"

RUN python -m pip check

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