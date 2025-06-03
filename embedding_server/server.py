
import uvicorn
import asyncio
import torch
from fastapi import FastAPI, Request, HTTPException, Query
from typing import List
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI()

supported_models = {
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

loaded_models = {}


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/load-model")
def load_models(models: List[str] = Query(...)):
    """Load one or more embedding models into memory"""

    unsupported = [
        m for m in models if m not in supported_models
    ]

    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model name(s): {', '.join(unsupported)}"
        )

    for name in models:

        if name in loaded_models:
            print(f"Model '{name}' already loaded.", flush=True)
            continue

        try:
            path = supported_models[name]
            print(f"→ Loading embedding model: {name} ({path}) ...", flush=True)
            loaded_models[name] = SentenceTransformer(path, device="cuda", trust_remote_code=True)
            print(f"  ✓ Model '{name}' loaded successfully.", flush=True)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model '{name}': {e}"
            )

    return { "success": True }


@app.delete("/unload-model/{model_name}")
def unload_model(model_name: str):
    """Unload a single embedding model from memory"""

    if model_name not in loaded_models:
        return {
            "success": True, "unloaded": model_name
        }

    del loaded_models[model_name]
    torch.cuda.empty_cache()  # only needed if using CUDA

    return {
        "success": True, "unloaded": model_name
    }


@app.delete("/unload-all-models")
def unload_all_models():
    """Unload all embedding models from memory"""

    if model_name not in loaded_models:
        del loaded_models[model_name]
        torch.cuda.empty_cache()  # only needed if using CUDA

    return {
        "success": True, "unloaded": model_name
    }


@app.get("/models")
def list_models():
    """List all available model names"""

    return list(loaded_models.keys())


@app.get("/vector-sizes")
def vector_sizes():
    """Return embedding size for each model"""

    return {
        name: model.get_sentence_embedding_dimension()
        for name, model in loaded_models.items()
    }


@app.get("/max-tokens")
def max_tokens():
    """Return maximum supported token length for each embedding model"""

    max_lengths = {}
    for name in loaded_models:
        path = supported_models[name]
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

    if embed_model not in loaded_models:
        raise HTTPException(status_code=400, detail="Embedding model not loaded.")

    model = loaded_models[embed_model]

    # Offload encoding to thread pool
    model_encode = await asyncio.get_event_loop().run_in_executor(
        executor, model.encode, texts
    )

    return { "embeddings": model_encode.tolist() }


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=5005)
