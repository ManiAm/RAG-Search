
import os
import sys
from typing import Dict, Optional
from datetime import datetime
from queue import Queue

from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, Query
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_unstructured import UnstructuredLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.document import Document

import config
from services.ollama_models import OllamaModels
from services.qdrant_db import Qdrant_DB
from services.streaming import token_generator, QueueCallbackHandler

ollama_obj = OllamaModels(config.ollama_url)
if not ollama_obj.check_health():
    sys.exit(1)

qdrant_obj = Qdrant_DB(qdrant_url=config.qdrant_url,
                       embedding_url=config.embedding_url)

os.makedirs(config.upload_dir, exist_ok=True)

# Session memory store (in-memory)
session_memories = {}
session_histories: Dict[str, ChatMessageHistory] = {}

app = APIRouter(prefix="/rag", tags=["rag"])

class CollectionRequest(BaseModel):
    collection_name: str
    embed_model: str


class ChatRequest(BaseModel):
    llm_model: str
    embed_model: str
    collection_name: str
    question: str
    session_id: Optional[str] = None
    instructions: Optional[str] = None


@app.get("/embeddings")
def get_embed_models():

    models = qdrant_obj.list_models()
    return {"models": models}


@app.get("/debug-search")
def debug_search(
    query: str = Query(..., description="Your semantic search query"),
    embed_model: str = Query(..., description="The embedding model used"),
    collection_name: str = Query(..., description="The Collection name"),
    k: int = Query(5, description="Number of top results to return")
):

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: /debug-search called with query='{query}', embed_model='{embed_model}', k={k}")

    status, output = qdrant_obj.get_retriever(embed_model, collection_name, k=k)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    retriever = output

    try:
        matches = retriever.invoke(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return just the page content and optional metadata
    return {
        "query": query,
        "results": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in matches
        ]
    }


@app.post("/create-collection")
def create_collection(req: CollectionRequest):

    embed_model = req.embed_model
    collection_name = req.collection_name

    status, output = qdrant_obj.create_collection(embed_model, collection_name)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    return {
        "status": "ok",
        "embed_model": embed_model,
        "collection_name": collection_name
    }


@app.post("/upload")
def upload_file(file: UploadFile = File(...), embed_model: str = Query(), collection_name: str = Query()):

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    filename = file.filename or "uploaded_file"
    collection_name = collection_name or embed_model

    print(f"{ts}: /upload endpoint called with filename '{filename}', embed_model '{embed_model}', collection_name '{collection_name}'.")

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: saving document...")

    file_path = os.path.join(config.upload_dir, filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: loading document...")

    loader = UnstructuredLoader(file_path)
    documents = loader.load()

    # Combine all small docs into one big document to improve chunking
    combined_text = "\n".join([doc.page_content for doc in documents])
    metadata = documents[0].metadata if documents else {}
    single_doc = Document(page_content=combined_text, metadata=metadata)
    documents = [single_doc]

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: splitting document...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Filter out metadata types that are not supported for a vector store.
    filtered_chunks = filter_complex_metadata(chunks)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: chunk count: {len(filtered_chunks)}")

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: embedding document...")

    status, output = qdrant_obj.add_documents(embed_model,
                                              collection_name,
                                              filtered_chunks)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: all done!")

    return {
        "status": "ok",
        "embed_model": embed_model,
        "collection_name": collection_name,
        "file": file.filename
    }


@app.post("/paste")
def paste_text(req: Dict[str, str]):

    text = req.get("text", "").strip()
    embed_model = req.get("embed_model", "")
    collection_name = req.get("collection_name", "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")

    if not embed_model:
        raise HTTPException(status_code=400, detail="No embed_model provided.")

    collection_name = collection_name or embed_model

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: /paste endpoint called with embed_model '{embed_model}'.")

    # Convert to Document object
    doc = Document(page_content=text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([doc])

    filtered_chunks = filter_complex_metadata(chunks)

    status, output = qdrant_obj.add_documents(embed_model,
                                              collection_name,
                                              filtered_chunks)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    return {
        "status": "ok",
        "embed_model": embed_model,
        "collection_name": collection_name,
        "chunks": len(filtered_chunks)
    }


@app.post("/chat")
def chat(req: ChatRequest):

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    llm_model = req.llm_model
    embed_model = req.embed_model
    collection_name = req.collection_name or embed_model
    session_id = req.session_id or "default"
    instructions = req.instructions or ""
    question = req.question

    print(f"{ts}: /chat endpoint called with llm_model '{llm_model}', embed_model '{embed_model}', collection_name '{collection_name}', session_id '{session_id}'")

    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()

    status, output = qdrant_obj.get_retriever(embed_model, collection_name, k=15)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    retriever = output

    ############

    print("")

    print("====== Instructions ======")
    print(f"{instructions.strip()}")
    print("=====================")

    print("")

    retrieved_docs = retriever.invoke(question)

    retrieved_docs_as_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    print("====== Context ======")
    print(f"{retrieved_docs_as_text.strip()}")
    print("=====================")

    print("")

    ############

    custom_prompt = PromptTemplate.from_template("""
        {instructions}

        Context:
        {context}

        Question:
        {question}

        Answer:
    """)

    if not ollama_obj.is_available(llm_model):
        raise HTTPException(status_code=400, detail=f"LLM model '{llm_model}' not loaded.")

    llm = OllamaLLM(model=llm_model,
                    base_url=config.ollama_url,
                    num_ctx=9000,
                    temperature=0.5)

    # This will inject the retrieved documents into {context} in the prompt
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                     retriever=retriever,
                                                     combine_docs_chain_kwargs={"prompt": custom_prompt})

    chain_with_memory = RunnableWithMessageHistory(
        qa_chain,
        lambda session_id: session_histories[session_id],
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    response = chain_with_memory.invoke(
        {
            "question": question,
            "instructions": instructions
        },
        config={"configurable": {"session_id": session_id}}
    )

    print("Response is ready!")

    return { "answer": response["answer"] }


@app.post("/chat-stream")
def chat_stream(req: ChatRequest):

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    llm_model = req.llm_model
    embed_model = req.embed_model
    collection_name = req.collection_name or embed_model
    session_id = req.session_id or "default"
    instructions = req.instructions or ""
    question = req.question

    print(f"{ts}: /chat-stream endpoint called with llm_model '{llm_model}', embed_model '{embed_model}', collection_name '{collection_name}', session_id '{session_id}'")

    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()

    status, output = qdrant_obj.get_retriever(embed_model, collection_name, k=15)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    retriever = output

    ############

    print("")

    print("====== Instructions ======")
    print(f"{instructions.strip()}")
    print("=====================")

    print("")

    retrieved_docs = retriever.invoke(question)

    retrieved_docs_as_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    print("====== Context ======")
    print(f"{retrieved_docs_as_text.strip()}")
    print("=====================")

    print("")

    ############

    custom_prompt = PromptTemplate.from_template("""
        {instructions}

        Context:
        {context}

        Question:
        {question}

        Answer:
    """)

    q = Queue()
    handler = QueueCallbackHandler(queue=q)

    if not ollama_obj.is_available(llm_model):
        raise HTTPException(status_code=400, detail=f"LLM model '{llm_model}' not loaded.")

    # Streaming-capable LLM
    llm = OllamaLLM(model=llm_model,
                    base_url=config.ollama_url,
                    num_ctx=9000,
                    temperature=0.5,
                    callbacks=[handler])

    # This will inject the retrieved documents into {context} in the prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    chain_with_memory = RunnableWithMessageHistory(
        qa_chain,
        lambda session_id: session_histories[session_id],
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    def run_chain():

        try:
            chain_with_memory.invoke(
                {
                    "question": question,
                    "instructions": instructions
                },
                config={"configurable": {"session_id": session_id}}
            )
        except Exception as e:
            q.put(f"\n[ERROR] {str(e)}")
            q.put(None)

    return StreamingResponse(token_generator(run_chain, q),
                             media_type="text/plain",
                             headers={"X-Accel-Buffering": "no"})
