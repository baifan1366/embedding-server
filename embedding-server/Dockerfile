# 使用干净的基础镜像，自己安装最新版 PyTorch
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# 安装 PyTorch >= 2.6
RUN pip install torch==2.6.0+cpu torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
