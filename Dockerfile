FROM python:3.10-slim

WORKDIR /app

# Install the minimal native toolchain needed by llama-cpp-python when a wheel
# is unavailable for the current platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Prefer a prebuilt CPU wheel, but keep the compiler available as a fallback.
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.20" \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Install the remaining runtime dependencies.
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "app.py"]
