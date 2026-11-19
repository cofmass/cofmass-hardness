"""FastAPI 路由：health / chat / skills / docs / history / stream(SSE) / 删除治理。

阶段二：user_id/tenant_id 由 api.deps.ACLContext 统一下推；阶段一请求体契约保持兼容。
阶段三：/chat/stream 从 501 改为真实 SSE（token / tool_start / tool_end / done）。
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import storage as store
from api.deps import ACLContext, acl_from_request
from api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamRequest,
    DocIngestRequest,
    SkillsPutRequest,
    SkillsResponse,
)
from config import API_VERSION, EMB_MODEL, SSE_PING_INTERVAL
from graph import nodes
from graph.async_builder import make_async_agent
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
    acl = acl_from_request(request, body_user_id=req.user_id,
                           body_tenant_id=getattr(req, "tenant_id", None))
    return agent.invoke(req.question, user_id=acl.user_id,
                        thread_id=req.thread_id, tenant_id=acl.tenant_id)


@router.get("/skills", response_model=SkillsResponse)
def skills_get(request: Request, user_id: str = "u-1", tenant_id: str | None = None) -> dict:
    agent = _agent(request)
    acl = acl_from_request(request, body_user_id=user_id, body_tenant_id=tenant_id)
    return nodes.get_skills(
        acl.user_id,
        agent.store,  # SQL 门面（mem_get 回退路径）
        agent.conn,
        tenant_id=acl.tenant_id,
        langgraph_store=agent.langgraph_store,  # BaseStore（PGStore）优先
    )


@router.put("/skills")
def skills_put(req: SkillsPutRequest, request: Request) -> dict:
    agent = _agent(request)
    acl = acl_from_request(request, body_user_id=req.user_id,
                           body_tenant_id=getattr(req, "tenant_id", None))
    nodes.set_skills(
        acl.user_id,
        agent.store,  # SQL 门面（mem_put 回退路径）
        agent.conn,
        preferences=req.preferences,
        enabled=req.enabled,
        tenant_id=acl.tenant_id,
        langgraph_store=agent.langgraph_store,  # BaseStore（PGStore）优先
    )
    return {"ok": True}


@router.post("/docs/ingest")
def docs_ingest(req: DocIngestRequest, request: Request) -> dict:
    agent = _agent(request)
    acl = acl_from_request(request, body_user_id=req.user_id)
    vec = get_embeddings().embed_query(req.content)
    metadata = dict(req.metadata or {})
    metadata.setdefault("user_id", acl.user_id)
    if acl.tenant_id:
        metadata.setdefault("tenant_id", acl.tenant_id)
    doc_id = agent.store.doc_add(
        agent.conn,
        req.content,
        vec,
        EMB_MODEL,
        metadata=metadata,
        source=req.source,
    )
    return {"id": doc_id}


@router.delete("/docs/{doc_id}")
def docs_delete(doc_id: int, request: Request, user_id: str = "u-1",
                tenant_id: str | None = None) -> dict:
    agent = _agent(request)
    acl = acl_from_request(request, body_user_id=user_id, body_tenant_id=tenant_id)
    deleted = agent.store.doc_delete(
        agent.conn, doc_id,
        user_id=acl.user_id, tenant_id=acl.tenant_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="文档不存在或不属于当前 ACL")
    return {"ok": True, "id": doc_id}


@router.delete("/memories")
def memories_delete(request: Request, key: str, user_id: str = "u-1",
                    tenant_id: str | None = None) -> dict:
    agent = _agent(request)
    acl = acl_from_request(request, body_user_id=user_id, body_tenant_id=tenant_id)
    namespace = acl.namespace("facts")
    if agent.langgraph_store is not None:
        agent.langgraph_store.delete(namespace, key)
    else:
        agent.store.mem_delete(
            agent.conn,
            __import__("json").dumps(list(namespace), ensure_ascii=False),
            key,
        )
    return {"ok": True}


@router.get("/chat/history")
def chat_history(request: Request, thread_id: str = "t-1", user_id: str = "u-1",
                 tenant_id: str | None = None) -> dict:
    agent = _agent(request)
    acl = acl_from_request(request, body_user_id=user_id, body_tenant_id=tenant_id)
    steps = agent.history(thread_id)
    return {"steps": steps, "thread_id": thread_id,
            "user_id": acl.user_id, "tenant_id": acl.tenant_id}


@router.post("/chat/stream")
def chat_stream(req: ChatStreamRequest, request: Request):
    """SSE 流式对话：token → tool_start/tool_end → done（阶段三）。"""
    agent = _agent(request)
    acl = acl_from_request(request, body_user_id=req.user_id,
                           body_tenant_id=req.tenant_id)
    async_agent = make_async_agent(agent)
    request_id = request.state.request_id

    done = object()

    async def gen():
        def sse(event: str, payload: dict) -> bytes:
            data = json.dumps({"event": event, "payload": payload},
                              ensure_ascii=False, separators=(",", ":"))
            return f"event: {event}\ndata: {data}\n\n".encode("utf-8")

        queue: asyncio.Queue = asyncio.Queue()

        async def pump():
            try:
                async for ev in async_agent.stream_events(
                    req.question, user_id=acl.user_id,
                    thread_id=req.thread_id, tenant_id=acl.tenant_id,
                ):
                    await queue.put(sse(ev["event"], ev["payload"]))
            except Exception as exc:  # 流内异常：发 error 事件后结束
                await queue.put(sse("error", {
                    "code": "STREAM_ERROR", "message": str(exc),
                    "request_id": request_id,
                }))
            finally:
                await async_agent.aclose()
                await queue.put(done)

        task = asyncio.create_task(pump())
        ping_every = max(SSE_PING_INTERVAL, 1.0)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=ping_every)
                except asyncio.TimeoutError:
                    yield b": ping\n\n"
                    continue
                if item is done:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "x-request-id": request_id},
    )
