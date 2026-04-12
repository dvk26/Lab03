FROM python:3.10-bookworm

WORKDIR /app

# uid 1000 is required by HF Spaces Docker runtime
RUN useradd -m -u 1000 user && \
    mkdir -p /app/models /app/artifacts && \
    chown -R 1000:1000 /app /home/user

# llama-cpp-python: binary wheel only; never compile from source on the HF builder.
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.19" \
    --only-binary=llama-cpp-python \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Pin runtime deps to avoid resolver drift on HF's Python 3.10 image.
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

COPY --chown=1000:1000 app.py .
COPY --chown=1000:1000 lab03/ ./lab03/
# artifacts/ only contains .gitkeep right now — COPY of an empty dir fails on HF BuildKit.
# The directory is already created above by mkdir -p /app/artifacts.
# Once you build and commit real artifact files, add: COPY --chown=1000:1000 artifacts/ ./artifacts/

USER 1000

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app

CMD ["python", "app.py"]
