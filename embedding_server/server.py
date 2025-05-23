
from fastapi import FastAPI, Request
from fastapi import HTTPException
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI()

models = {
    "all-MiniLM-L6-v2": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "all-MiniLM-L12-v2": SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2"),
    "bge-base-en-v1.5": SentenceTransformer("BAAI/bge-base-en-v1.5"),
    "bge-large-en-v1.5": SentenceTransformer("BAAI/bge-large-en-v1.5")
}


@app.get("/health")
def health():
    return {"status": "running"}


@app.get("/models")
def list_models():
    """List all available model names"""

    return list(models.keys())


@app.get("/vector-sizes")
def vector_sizes():
    """Return embedding size for each model"""

    vec_size = {
        name: model.get_sentence_embedding_dimension() for name, model in models.items()
    }
    return vec_size


@app.post("/embed")
async def embed(request: Request):

    body = await request.json()

    texts = body.get("texts", [])
    embed_model = body.get("model", "")

    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise HTTPException(status_code=400, detail="'texts' must be a list of strings")

    if embed_model not in models:
        raise HTTPException(status_code=400, detail="Embedding model not loaded.")

    model = models[embed_model]

    # Offload to thread pool
    model_encode = await asyncio.get_event_loop().run_in_executor(
        executor, model.encode, texts
    )

    return {"embeddings": model_encode.tolist()}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=5005)
