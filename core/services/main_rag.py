
import os
import sys
import json
import pprint
from typing import Dict, Optional, Any, List
from datetime import datetime
from queue import Queue

from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, Query
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_unstructured import UnstructuredLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema.document import Document

import config
from services.lite_llm import LiteLLM
from services.remote_embedding import RemoteEmbedding
from services.qdrant_db import Qdrant_DB
from services.streaming import token_generator, QueueCallbackHandler

litellm_obj = LiteLLM(config.lite_llm_url)
if not litellm_obj.is_reachable():
    sys.exit(1)

qdrant_obj = Qdrant_DB(qdrant_url=config.qdrant_url, embedding_url=config.embedding_url)

embedding_models_to_load = [
    "bge-large-en-v1.5"
]
print(f"Loading embedding models: {embedding_models_to_load}")
status, output = qdrant_obj.load_model(embedding_models_to_load)
if not status:
    print(f"Error: {output}")
    sys.exit(1)

os.makedirs(config.upload_dir, exist_ok=True)

# Session memory store (in-memory)
session_memories = {}
session_histories: Dict[str, ChatMessageHistory] = {}

app = APIRouter(prefix="/rag", tags=["rag"])

class CollectionRequest(BaseModel):
    collection_name: str
    embed_model: str


class DeleteByFilterRequest(BaseModel):
    collection_name: str
    filter: Dict[str, Any]


class PasteRequest(BaseModel):
    text: str
    embed_model: str
    collection_name: Optional[str] = None
    metadata: Optional[str] = None
    separators: Optional[List[str]] = None
    chunk_size: Optional[int] = None


class SplitDocument(BaseModel):
    text: str
    separators: Optional[List[str]] = None
    chunk_size: Optional[int] = None


class EmbedTextRequest(BaseModel):
    text: str
    embed_model: str
    separators: Optional[List[str]] = None
    chunk_size: Optional[int] = None


class AddPointsRequest(BaseModel):
    embed_model: str
    collection_name: str
    vectors: List[List[float]]
    texts: Optional[List[str]] = None
    metadata: Optional[Dict] = None


class ChatRequest(BaseModel):
    llm_model: str
    embed_model: str
    collection_name: str
    question: str
    session_id: Optional[str] = None
    instructions: Optional[str] = None
    score_threshold: Optional[float] = None
    max_documents: Optional[int] = None

###########################

@app.post("/load-model")
def load_model(models: List[str] = Query(...)):

    status, output = qdrant_obj.load_model(models)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    return { "success": True }


@app.delete("/unload-model/{model_name}")
def unload_model(model_name: str):

    status, output = qdrant_obj.unload_model(model_name)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    return { "success": True }


@app.delete("/unload-all-models")
def unload_all_models():

    status, output = qdrant_obj.unload_all_models()
    if not status:
        raise HTTPException(status_code=400, detail=output)

    return { "success": True }

###########################

@app.get("/embeddings")
def get_embed_models():

    models = qdrant_obj.list_models()
    return {"models": models}


@app.get("/max-tokens")
def get_model_max_tokens():

    max_tokens_map = qdrant_obj.get_model_max_token()
    return max_tokens_map


@app.get("/collections")
def list_collections():

    return qdrant_obj.list_collections()


@app.post("/create-collection")
def create_collection(req: CollectionRequest):

    embed_model = req.embed_model
    collection_name = req.collection_name

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: /create-collection called with collection_name='{collection_name}', embed_model='{embed_model}'")

    status, output = qdrant_obj.create_collection(embed_model, collection_name)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    return {
        "status": "ok",
        "embed_model": embed_model,
        "collection_name": collection_name
    }

###########################

@app.delete("/del-by-filter")
def delete_by_filter(req: DeleteByFilterRequest):

    collection = req.collection_name.strip()
    filter_dict = req.filter

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: /del-by-filter endpoint called with collection '{collection}', filter: {filter_dict}")

    if not collection or not filter_dict:
        raise HTTPException(status_code=400, detail="Missing required fields.")

    try:
        delete_response  = qdrant_obj.delete_points_by_filter(collection, filter_dict)

        return {
            "status": "ok",
            "collection_name": collection,
            "filter": filter_dict,
            "qdrant_response": str(delete_response.status)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug-search")
def debug_search(
    query: str = Query(..., description="Your semantic search query"),
    embed_model: str = Query(..., description="The embedding model used"),
    collection_name: str = Query(..., description="The Collection name"),
    k: int = Query(5, description="Number of top results to return")):

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: /debug-search called with query='{query}', embed_model='{embed_model}', k={k}")

    status, output = qdrant_obj.get_store(embed_model, collection_name, create_collection=False)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    qdrant_store = output

    retriever = qdrant_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

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

###########################

@app.post("/upload")
def upload_file(file: UploadFile = File(...),
                embed_model: str = Query(),
                collection_name: str = Query(),
                chunk_size: Optional[int] = None,
                separators: Optional[List[str]] = None):

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    filename = file.filename or "uploaded_file"
    collection_name = collection_name or embed_model
    chunk_size = chunk_size or 1000
    separators = separators or ["\n\n", "\n", " ", ""]

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

    check_chunk_size(chunk_size, embed_model)

    doc_chunks = split_document(documents, chunk_size, separators)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: chunk count: {len(doc_chunks)}")

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: embedding document...")

    status, output = qdrant_obj.add_documents(embed_model,
                                              collection_name,
                                              doc_chunks)
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
def paste_text(req: PasteRequest):

    text = req.text.strip()
    embed_model = req.embed_model.strip()
    collection_name = (req.collection_name or embed_model).strip()
    metadata = json.loads(req.metadata) if req.metadata else {}
    chunk_size = req.chunk_size or 1000
    separators = req.separators or ["\n\n", "\n", " ", ""]

    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")

    if not embed_model:
        raise HTTPException(status_code=400, detail="No embed_model provided.")

    collection_name = collection_name or embed_model

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: /paste endpoint called with embed_model '{embed_model}'.")

    # Convert to Document object
    doc = Document(page_content=text, metadata=metadata)

    check_chunk_size(chunk_size, embed_model)

    doc_chunks = split_document([doc], chunk_size, separators)
    chunk_texts = [chunk.page_content for chunk in doc_chunks]

    status, output = qdrant_obj.add_documents(embed_model,
                                              collection_name,
                                              doc_chunks)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    return {
        "status": "ok",
        "embed_model": embed_model,
        "chunk_text": chunk_texts,
        "chunks": len(doc_chunks),
        "collection_name": collection_name
    }

###########################

@app.post("/embed_text")
def embedd_text(req: EmbedTextRequest):
    """
    Tokenizes the input text into smaller chunks and computes an embedding vector for each chunk
    using the specified embedding model.
    """

    text = req.text.strip()
    embed_model = req.embed_model.strip()
    chunk_size = req.chunk_size or 1000
    separators = req.separators or ["\n\n", "\n", " ", ""]

    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")

    if not embed_model:
        raise HTTPException(status_code=400, detail="No embed_model provided.")

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: /embed_text endpoint called with embed_model '{embed_model}'.")

    # Convert to Document object
    doc = Document(page_content=text)

    check_chunk_size(chunk_size, embed_model)

    doc_chunks = split_document([doc], chunk_size, separators)
    chunk_texts = [chunk.page_content for chunk in doc_chunks]

    embedding_model = RemoteEmbedding(endpoint=f"{config.embedding_url}/embed", model=embed_model)

    try:
        vectors = embedding_model.embed_documents(chunk_texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    return {
        "status": "ok",
        "embed_model": embed_model,
        "chunk_text": chunk_texts,
        "vectors": vectors
    }


@app.post("/add_points")
def add_points(req: AddPointsRequest):
    """
    Insert precomputed embeddings into a Qdrant collection with optional shared metadata.
    """

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts}: /add_points endpoint called.")

    status, error = qdrant_obj.add_points(
        embed_model=req.embed_model,
        collection_name=req.collection_name,
        vectors=req.vectors,
        texts=req.texts,
        metadata=req.metadata
    )

    if not status:
        raise HTTPException(status_code=500, detail=error)

    return {
        "status": "success",
        "message": "Embeddings added to Qdrant successfully."
    }

###########################

@app.post("/split-doc")
def split_doc(req: SplitDocument):

    text = req.text.strip()
    chunk_size = req.chunk_size or 1000
    separators = req.separators or ["\n\n", "\n", " ", ""]

    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")

    # Convert to Document object
    doc = Document(page_content=text)

    doc_chunks = split_document([doc], chunk_size, separators)

    chunks_text = [chunk.page_content for chunk in doc_chunks]

    return JSONResponse(content={"chunks": chunks_text, "count": len(chunks_text)})


def split_document(documents, chunk_size, separators):

    splitter = safe_recursive_splitter(chunk_size, 200, separators)
    chunks = splitter.split_documents(documents)

    # Remove chunks that contain any of the separators
    filtered = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        any_matched = any(sep == content for sep in separators)
        if not any_matched:
            filtered.append(chunk)

    # Filter out metadata types that are not supported for a vector store.
    filtered_chunks = filter_complex_metadata(filtered)

    return filtered_chunks


def safe_recursive_splitter(chunk_size: int, chunk_overlap: int, separators: list):

    # Ensure chunk_overlap is valid
    if chunk_overlap >= chunk_size:
        # Automatically reduce to 25% of chunk_size or less
        chunk_overlap = max(0, chunk_size // 4)

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )


def check_chunk_size(chunk_size, embed_model):

    max_tokens_map = qdrant_obj.get_model_max_token()
    if not max_tokens_map:
        return

    if not isinstance(max_tokens_map, dict):
        return

    max_tokens = max_tokens_map.get(embed_model, None)
    if not max_tokens:
        return

    avg_chars_per_token = 3.5  # in English
    approx_max_characters = max_tokens * avg_chars_per_token

    if chunk_size > approx_max_characters:
        print(f"Warning: chunk_size={chunk_size} is bigger than maximum token size of embedding mode {embed_model}")

###########################

@app.post("/chat")
def chat(req: ChatRequest):

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    llm_model = req.llm_model
    embed_model = req.embed_model
    collection_name = req.collection_name or embed_model
    session_id = req.session_id or "default"
    instructions = req.instructions or ""
    question = req.question
    score_threshold = req.score_threshold or 0.7
    max_documents = req.max_documents or 5

    print(f"{ts}: /chat endpoint called with llm_model '{llm_model}', embed_model '{embed_model}', collection_name '{collection_name}', session_id '{session_id}'")

    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()

    status, output = qdrant_obj.get_store(embed_model, collection_name, create_collection=False)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    qdrant_store = output

    retriever = qdrant_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": score_threshold,
            "k": max_documents
        }
    )

    ############

    print("")

    print("============= Question =============")
    print(f"{question.strip()}")
    print("===================================")

    print("")

    docs_with_scores = qdrant_store.similarity_search_with_relevance_scores(
        question,
        score_threshold=score_threshold,
        k=max_documents
    )

    print_retrieved_docs_table(docs_with_scores)

    ############

    print("")

    print("====== Instructions ======")
    print(f"{instructions.strip()}")
    print("==========================")

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

    if not litellm_obj.is_available(llm_model):
        raise HTTPException(status_code=400, detail=f"LLM model '{llm_model}' not loaded.")

    llm = ChatOpenAI(base_url=config.lite_llm_url,
                     model=llm_model,
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

    ############

    print("")

    print("======== Response ========")
    print(response["answer"].strip())
    print("==========================")

    print("")

    ############

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
    score_threshold = req.score_threshold or 0.7
    max_documents = req.max_documents or 5

    print(f"{ts}: /chat-stream endpoint called with llm_model '{llm_model}', embed_model '{embed_model}', collection_name '{collection_name}', session_id '{session_id}'")

    # streaming does not work properly with memory
    # if session_id not in session_histories:
    session_histories[session_id] = ChatMessageHistory()

    status, output = qdrant_obj.get_store(embed_model, collection_name, create_collection=False)
    if not status:
        raise HTTPException(status_code=400, detail=output)

    qdrant_store = output

    retriever = qdrant_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": score_threshold,
            "k": max_documents
        }
    )

    ############

    print("")

    print("============= Question =============")
    print(f"{question.strip()}")
    print("===================================")

    print("")

    docs_with_scores = qdrant_store.similarity_search_with_relevance_scores(
        question,
        score_threshold=score_threshold,
        k=max_documents
    )

    print_retrieved_docs_table(docs_with_scores)

    ############

    print("")

    print("====== Instructions ======")
    print(f"{instructions.strip()}")
    print("==========================")

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

    if not litellm_obj.is_available(llm_model):
        raise HTTPException(status_code=400, detail=f"LLM model '{llm_model}' not loaded.")

    # Streaming-capable LLM
    llm = ChatOpenAI(base_url=config.lite_llm_url,
                     model=llm_model,
                     temperature=0.5,
                     callbacks=[handler],
                     streaming=True)

    # This will inject the retrieved documents into {context} in the prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=False
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
        finally:
            q.put(None)

    return StreamingResponse(token_generator(run_chain, q),
                             media_type="text/plain",
                             headers={"X-Accel-Buffering": "no"})


def print_retrieved_docs_table(docs_with_scores):

    print(f"Found {len(docs_with_scores)} documents.")

    docs_with_scores.sort(key=lambda x: x[1], reverse=True)

    for idx, (doc, score) in enumerate(docs_with_scores):

        print(f"\nDocument {idx+1}")

        print(f"Score: {score}")

        print("\nMetadata:\n")
        metadata = doc.metadata
        pretty = pprint.pformat(metadata)
        indented = "\n".join(" "*5 + line for line in pretty.splitlines())
        print(indented)

        print("\nContent:\n")
        doc_txt = doc.page_content
        indented = "\n".join(" "*5 + line for line in doc_txt.splitlines())
        print(indented)
