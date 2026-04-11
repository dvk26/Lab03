FROM python:3.10-slim

WORKDIR /app

# Cài build tools nhẹ (chỉ cần cho một số C extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Cài llama-cpp-python từ pre-built wheel TRƯỚC (tránh OOM khi compile)
RUN pip install --no-cache-dir \
    llama-cpp-python==0.3.9 \
    --prefer-binary \
    --find-links https://abetlen.github.io/llama-cpp-python/whl/cpu

# Cài phần còn lại
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "app.py"]
