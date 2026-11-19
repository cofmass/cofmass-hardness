"""FastAPI 应用：lifespan 单例 + 统一错误结构 + CORS。"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import storage as store
from api.routes import router
from config import API_CORS_ORIGINS, API_VERSION
from graph.builder import Agent
from providers.factory import get_chat_model, get_embeddings
from storage.checkpointer import make_saver

log = logging.getLogger("hardness.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用级长生命周期连接 + saver；请求内禁止 with 包裹或关闭连接。
    conn = store.connect()
    store.setup(conn)
    saver, _ = make_saver()
    app.state.agent = Agent(
        conn,
        saver,
        llm=get_chat_model(),
        embeddings=get_embeddings(),
        storage_module=store,
    )
    log.info("agent ready: backend=%s", store.backend_name())
    try:
        yield
    finally:
        app.state.agent.close()
        app.state.agent = None


app = FastAPI(
    title="Hardness Agent API",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CORS_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.middleware("http")
async def request_id_and_errors(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception as exc:  # 统一错误体
        log.exception("unhandled error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                    "request_id": request_id,
                }
            },
        )
    response.headers["x-request-id"] = request_id
    return response
