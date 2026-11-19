"""FastAPI 路由：health / chat / skills / docs / history / stream(501)。"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request

import storage as store
from api.schemas import (
    ChatRequest,
    ChatResponse,
    DocIngestRequest,
    SkillsPutRequest,
    SkillsResponse,
)
from config import API_VERSION, EMB_MODEL
from graph import nodes
from providers.factory import get_embeddings

router = APIRouter()


def _agent(request: Request):
    agent = request.app.state.agent
    if agent is None:
        raise HTTPException(status_code=503, detail="agent 未就绪")
    return agent


def _error(code: str, message: str, request_id: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": API_VERSION}


@router.post("/chat/invoke", response_model=ChatResponse)
def chat_invoke(req: ChatRequest, request: Request) -> dict:
    agent = _agent(request)
    return agent.invoke(req.question, user_id=req.user_id, thread_id=req.thread_id)


@router.get("/skills", response_model=SkillsResponse)
def skills_get(user_id: str = "u-1", request: Request = None) -> dict:
    agent = _agent(request)
    return nodes.get_skills(user_id, agent.store, agent.conn)


@router.put("/skills")
def skills_put(req: SkillsPutRequest, request: Request) -> dict:
    agent = _agent(request)
    nodes.set_skills(
        req.user_id,
        agent.store,
        agent.conn,
        preferences=req.preferences,
        enabled=req.enabled,
    )
    return {"ok": True}


@router.post("/docs/ingest")
def docs_ingest(req: DocIngestRequest, request: Request) -> dict:
    agent = _agent(request)
    vec = get_embeddings().embed_query(req.content)
    metadata = dict(req.metadata or {})
    metadata.setdefault("user_id", req.user_id)
    doc_id = agent.store.doc_add(
        agent.conn,
        req.content,
        vec,
        EMB_MODEL,
        metadata=metadata,
        source=req.source,
    )
    return {"id": doc_id}


@router.get("/chat/history")
def chat_history(thread_id: str = "t-1", user_id: str = "u-1", request: Request = None) -> dict:
    agent = _agent(request)
    steps = agent.history(thread_id)
    # 阶段一 user_id 校验：history 中的消息由该 thread 产生；归属校验留给阶段二 deps 统一处理。
    return {"steps": steps, "thread_id": thread_id, "user_id": user_id}


@router.post("/chat/stream")
def chat_stream(request: Request):
    request_id = str(uuid.uuid4())
    return _error(
        "NOT_IMPLEMENTED",
        "SSE 流式在阶段三启用；阶段一仅登记契约",
        request_id,
    )
