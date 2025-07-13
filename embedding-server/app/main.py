from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # multilingual model
DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("USE_CUDA", "false") == "true" else "cpu")

print(f"Loading model: {MODEL_NAME} on {DEVICE}")
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

class EmbeddingRequest(BaseModel):
    input: str

@app.post("/embed")
async def embed(request: EmbeddingRequest):
    embedding = model.encode(request.input, normalize_embeddings=True).tolist()
    return {
        "embedding": embedding,
        "model": MODEL_NAME,
        "dim": len(embedding)
    }

class BatchRequest(BaseModel):
    inputs: list[str]

@app.post("/embed/batch")
async def embed_batch(request: BatchRequest):
    vectors = model.encode(request.inputs, normalize_embeddings=True).tolist()
    return {
        "embeddings": vectors,
        "count": len(vectors),
        "model": MODEL_NAME,
        "dim": len(vectors[0]) if vectors else 0
    }
