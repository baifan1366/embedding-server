FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git wget ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 关键：设置 huggingface cache 路径
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HF_HOME=/app/hf_cache

# 给 cache 目录权限
RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

COPY app app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
