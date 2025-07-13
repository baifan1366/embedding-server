# 🧠 Local Embedding API Server (Multilingual)

A lightweight embedding generation server using FastAPI + SentenceTransformer, supporting Chinese, English, Malay and more — powered by `BAAI/bge-m3`.

## 🚀 Features

- `/embed`: generate embedding for a single string
- `/embed/batch`: generate embedding for multiple strings
- Support for GPU acceleration (CUDA) via Docker
- Docker-ready deployment

## 🔧 Requirements

- Docker (with NVIDIA container runtime)
- Optional: Python 3.10+ for local dev

## ⚙️ Run with Docker (GPU)

```bash
docker build -t embed-server .
docker run --gpus all --env-file .env -p 8000:8000 embed-server
```

## 📦 API Usage

### `POST /embed`
```json
{
  "input": "Saya suka makan nasi lemak"
}
```

### `POST /embed/batch`
```json
{
  "inputs": ["I like cats", "我喜欢猫", "Saya suka kucing"]
}
```

## 📤 Response

```json
{
  "embedding": [...],
  "dim": 384,
  "model": "BAAI/bge-m3"
}
```

## ✍️ License

MIT
