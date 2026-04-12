FROM python:3.10-bookworm

WORKDIR /app

# Hugging Face Docker Spaces work more reliably when the app runs as uid 1000.
RUN useradd -m -u 1000 user
RUN mkdir -p /app/models /app/artifacts && chown -R user:user /app /home/user

# llama-cpp-python prebuilt CPU wheel (no compilation, fast)
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.19" \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Runtime deps — fastembed replaces sentence-transformers+torch (ONNX, no torch needed)
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy only runtime-needed files — avoids pulling in storage/, models/, evaluation/, etc.
COPY --chown=user app.py .
COPY --chown=user pyproject.toml .
COPY --chown=user lab03/ ./lab03/
COPY --chown=user artifacts/ ./artifacts/

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

CMD ["python", "app.py"]
