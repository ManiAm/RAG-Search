
import uvicorn
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi import APIRouter

# Suppress info/debug logs from Qdrant or HTTPX
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import services.main_llm  # Direct LLM interaction
import services.main_rag  # RAG functionality

app = FastAPI(
    title="RAG-Search API",
    description="Retrieval-Augmented Generation chat with LLM and document support.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1 = APIRouter(prefix="/api/v1")
api_v1.include_router(services.main_llm.app)
api_v1.include_router(services.main_rag.app)
app.include_router(api_v1)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Open PowerShell or Command Prompt on Windows and run:
#   wsl hostname -I
# Now open your browser in Windows and go to:
#   172.29.198.1:8000

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


@app.get("/health", tags=["Health"])
def health():
    return {"status": "running"}


if __name__ == "__main__":

    print("Starting RAG-Search on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
