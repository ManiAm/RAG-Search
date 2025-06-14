FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    nano \
    curl \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

RUN pip install \
    langchain \
    langchain-core \
    langchain-community \
    langchain-openai \
    langchain-ollama \
    langchain-chroma \
    langchain-unstructured \
    langchain_text_splitters \
    langchain_qdrant \
    langgraph \
    openai \
    "unstructured[pdf]" \
    "unstructured[docx]" \
    faiss-gpu \
    "numpy<2" \
    qdrant-client \
    nltk \
    spacy \
    beautifulsoup4 \
    fpdf \
    jq \
    pymupdf

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
