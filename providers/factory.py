"""LLM / Embedding 工厂 —— 图与存储层只依赖这里返回的抽象接口。

切换 SDK 时只改本文件（或读配置），图/存储零改动。
阶段一用 Ollama；langchain-ollama 未装时自动降级到内置 HTTP 实现（仍走 ollama 服务）。
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import CHAT_MODEL, EMB_MODEL, OLLAMA_BASE_URL


# ---------- 本地轻量实现（无 langchain-ollama 时使用）----------

class _OllamaChatHTTP(BaseChatModel):
    """直接走 Ollama /api/chat，只实现最小 invoke 路径。"""

    model: str = CHAT_MODEL
    base_url: str = OLLAMA_BASE_URL
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "ollama-http"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessage
        # langchain-core >=0.3.70 把 ChatGeneration / ChatResult 移到 outputs，
        # messages 里只保留 AIMessage；兼容旧版优先从 messages 导入。
        try:
            from langchain_core.messages import ChatGeneration, ChatResult
        except ImportError:
            from langchain_core.outputs import ChatGeneration, ChatResult

        payload = {
            "model": self.model,
            "messages": [_to_ollama_msg(m) for m in messages],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        text = data.get("message", {}).get("content", "")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class _OllamaEmbeddingsHTTP(Embeddings):
    def __init__(self, model: str = EMB_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            payload = {"model": self.model, "prompt": text}
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.load(resp)
            vectors.append(data["embedding"])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def _to_ollama_msg(message: Any) -> dict:
    role = {"human": "user", "ai": "assistant", "system": "system"}.get(message.type, message.type)
    return {"role": role, "content": message.content}


# ---------- 工厂 ----------

def get_chat_model() -> BaseChatModel:
    """返回 ChatOllama（若安装 langchain-ollama），否则返回内置 HTTP 版。"""
    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(model=CHAT_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
    except ImportError:
        return _OllamaChatHTTP(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL)


def get_embeddings() -> Embeddings:
    """返回 OllamaEmbeddings（若安装 langchain-ollama），否则返回内置 HTTP 版。"""
    try:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=EMB_MODEL, base_url=OLLAMA_BASE_URL)
    except ImportError:
        return _OllamaEmbeddingsHTTP(model=EMB_MODEL, base_url=OLLAMA_BASE_URL)
