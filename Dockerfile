FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Step 1: install torch first alone (large wheel, isolate its resolution)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Step 2: install llama-cpp-python from prebuilt CPU wheel (avoids compile + OOM)
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.20" \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Step 3: install remaining runtime deps (no torch, no llama-cpp here)
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "app.py"]
