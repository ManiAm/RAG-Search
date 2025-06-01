
import uvicorn
import asyncio
from fastapi import FastAPI, Request, HTTPException
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI()

model_names = {
    "all-MiniLM-L6-v2"   : "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2"  : "sentence-transformers/all-MiniLM-L12-v2",

    "bge-base-en-v1.5"   : "BAAI/bge-base-en-v1.5",
    "bge-large-en-v1.5"  : "BAAI/bge-large-en-v1.5",
    "mxbai-embed-large"  : "mixedbread-ai/mxbai-embed-large-v1",
    "e5-base-v2"         : "intfloat/e5-base-v2",
    "e5-large-v2"        : "intfloat/e5-large-v2",
    "bge-m3"             : "BAAI/bge-m3",

    "nomic-embed-text"   : "nomic-ai/nomic-embed-text-v1"
}

models = {}

for name, path in model_names.items():

    print(f"→ Loading embedding model: {name} ({path}) ...", flush=True)
    try:
        models[name] = SentenceTransformer(path, trust_remote_code=True)
        print(f"  ✓ Model '{name}' loaded successfully.", flush=True)
    except Exception as e:
        print(f"  ✗ Failed to load model '{name}': {e}", flush=True)


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

    return {
        name: model.get_sentence_embedding_dimension()
        for name, model in models.items()
    }


@app.get("/max-tokens")
def max_tokens():
    """Return maximum supported token length for each embedding model"""

    max_lengths = {}
    for name, path in model_names.items():
        tokenizer = AutoTokenizer.from_pretrained(path)
        max_lengths[name] = tokenizer.model_max_length
    return max_lengths


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

    # Offload encoding to thread pool
    model_encode = await asyncio.get_event_loop().run_in_executor(
        executor, model.encode, texts
    )

    return { "embeddings": model_encode.tolist() }


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=5005)
