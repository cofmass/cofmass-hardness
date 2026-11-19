"""HTTP 契约（Pydantic v2）。user_id / thread_id 阶段一从请求体传入。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = "u-1"
    thread_id: str = "t-1"
    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    context: list[dict[str, Any]] = []
    skill_ctx: dict[str, Any] = {}
    thread_id: str
    step: int


class SkillsGetRequest(BaseModel):
    user_id: str = "u-1"


class SkillsPutRequest(BaseModel):
    user_id: str = "u-1"
    preferences: dict[str, Any] | None = None
    enabled: list[str] | None = None


class SkillsResponse(BaseModel):
    preferences: dict[str, Any] = {}
    enabled: dict[str, Any] = {}


class DocIngestRequest(BaseModel):
    user_id: str = "u-1"
    content: str = Field(min_length=1)
    source: str | None = None
    metadata: dict[str, Any] = {}


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
