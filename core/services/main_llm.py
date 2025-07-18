
import sys
from typing import Dict, Optional
from datetime import datetime
from queue import Queue

from pydantic import BaseModel
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query
from fastapi.responses import StreamingResponse

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import config
from services.lite_llm import LiteLLM
from services.streaming import token_generator, QueueCallbackHandler

litellm_obj = LiteLLM(config.lite_llm_url)
if not litellm_obj.is_reachable():
    sys.exit(1)

# Session memory store (in-memory)
session_memories = {}
session_histories: Dict[str, ChatMessageHistory] = {}

app = APIRouter(prefix="/llm", tags=["llm"])

class ChatRequest(BaseModel):
    question: str
    llm_model: str
    session_id: Optional[str] = None
    context: Optional[str] = None

#############################

@app.get("/models")
def get_llm_models():

    models = litellm_obj.list_models()
    return {"models": models}


@app.get("/model-info")
def get_model_info(model_name: str = Query(...)):

    info_map = litellm_obj.get_model_info(model_name)
    return info_map

#############################

@app.post("/chat")
def chat_llm(req: ChatRequest):

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    session_id = req.session_id or "default"
    question = req.question
    llm_model = req.llm_model
    context = req.context or ""

    print(f"{ts}: /chat-llm endpoint called with session_id '{session_id}', llm_model '{llm_model}', context '{context}'")

    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()

    ############

    custom_prompt = ChatPromptTemplate.from_template("""
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

    llm_chain = custom_prompt | llm

    chain_with_memory = RunnableWithMessageHistory(
        llm_chain,
        lambda sid: session_histories[sid],
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    result = chain_with_memory.invoke(
        {
            "question": question,
            "context": context
        },
        config={"configurable": {"session_id": session_id}}
    )

    if isinstance(result, AIMessage):
        return {"answer": result.content}

    raise HTTPException(status_code=400, detail=f"Was expecting an AIMessage.")



@app.post("/chat-stream")
def chat_stream(req: ChatRequest):

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    session_id = req.session_id or "default"
    question = req.question
    llm_model = req.llm_model
    context = req.context or ""

    print(f"{ts}: /chat-stream endpoint called with session_id '{session_id}', llm_model '{llm_model}', context '{context}'")

    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()

    ############

    custom_prompt = ChatPromptTemplate.from_template("""
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

    llm_chain = custom_prompt | llm

    chain_with_memory = RunnableWithMessageHistory(
        llm_chain,
        lambda sid: session_histories[sid],
        input_messages_key="question",
        history_messages_key="messages",
    )

    def run_chain():

        try:
            chain_with_memory.invoke(
                {
                    "question": question,
                    "context": context
                },
                config={"configurable": {"session_id": session_id}}
            )
        except Exception as e:
            q.put(f"\n[ERROR] {str(e)}")
            q.put(None)

    return StreamingResponse(token_generator(run_chain, q),
                             media_type="text/plain",
                             headers={"X-Accel-Buffering": "no"})
