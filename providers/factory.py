"""LLM / Embedding 工厂 —— 图与存储层只依赖这里返回的抽象接口。

阶段四：配置驱动（HARDNESS_PROVIDER），切换 Provider 只改配置，不改图/存储。
支持：ollama（默认，含 HTTP 降级）、openai / tongyi / claude（可选 SDK，缺失显式报错）、
fake（契约测试专用）。

红线：图/存储层不得 import 任何具体 SDK；SDK 只在工厂函数内部延迟导入。
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from config import CHAT_MODEL, EMB_MODEL, OLLAMA_BASE_URL, PROVIDER
from providers import config as pconfig


class UnknownProviderError(ValueError):
    """未知/不支持的 Provider 配置。"""


class MissingProviderSDKError(ImportError):
    """配置的 Provider SDK 未安装，禁止静默降级到错误模型。"""


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


class FakeChatModel(BaseChatModel):
    """契约测试专用：无网络、无 SDK，输出可预测（回显最后一条人消息）。"""

    model: str = "fake-chat"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            from langchain_core.messages import ChatGeneration, ChatResult
        except ImportError:
            from langchain_core.outputs import ChatGeneration, ChatResult

        text = "fake:" + (getattr(messages[-1], "content", "") or "") if messages else "fake:"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class FakeEmbeddings(Embeddings):
    """契约测试专用：确定性哈希向量，维度固定为配置的 EMB_DIM。"""

    def __init__(self, dim: int | None = None):
        from config import EMB_DIM
        self.dim = dim or EMB_DIM

    @staticmethod
    def _stable_vec(text: str, dim: int) -> list[float]:
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [b / 255.0 for b in digest]
        return [(raw[i % len(raw)] if i < len(raw) else 0.0) for i in range(dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._stable_vec(t, self.dim) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._stable_vec(text, self.dim)


def _to_ollama_msg(message: Any) -> dict:
    role = {"human": "user", "ai": "assistant", "system": "system"}.get(message.type, message.type)
    return {"role": role, "content": message.content}


# ---------- 工厂 ----------

def _require_sdk(module: str, package: str, provider: str) -> Any:
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as exc:
        raise MissingProviderSDKError(
            f"Provider {provider!r} 需要 {package}；请 pip install {package}"
        ) from exc


def get_chat_model(provider: str | None = None) -> BaseChatModel:
    """按配置返回 Chat 模型；图/存储零改动。"""
    p = (provider or PROVIDER).strip().lower()
    if p not in pconfig.CHAT_CONFIG:
        raise UnknownProviderError(f"未知 HARDNESS_PROVIDER={p!r}，可选：{sorted(pconfig.CHAT_CONFIG)}")
    cfg = pconfig.CHAT_CONFIG[p]

    if p == "fake":
        return FakeChatModel(model=cfg["model"])
    if p == "ollama":
        try:
            sdk = _require_sdk("langchain_ollama", "langchain-ollama", p)
            return sdk.ChatOllama(**cfg)
        except MissingProviderSDKError:
            # 阶段一既有行为：SDK 未装时走内置 HTTP（仍是 Ollama 服务，不换模型）。
            return _OllamaChatHTTP(model=cfg["model"], base_url=cfg["base_url"],
                                   temperature=cfg["temperature"])
    if p == "openai":
        sdk = _require_sdk("langchain_openai", "langchain-openai", p)
        return sdk.ChatOpenAI(model=cfg["model"], temperature=cfg["temperature"])
    if p == "tongyi":
        # 通义兼容 OpenAI 协议：用 ChatOpenAI + 自定义 base_url + DashScope key。
        sdk = _require_sdk("langchain_openai", "langchain-openai", p)
        return sdk.ChatOpenAI(
            model=cfg["model"], temperature=cfg["temperature"],
            base_url=os.environ.get("HARDNESS_TONGYI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        )
    if p == "claude":
        sdk = _require_sdk("langchain_anthropic", "langchain-anthropic", p)
        return sdk.ChatAnthropic(model=cfg["model"], temperature=cfg["temperature"])
    raise UnknownProviderError(p)  # pragma: no cover


def get_embeddings(provider: str | None = None) -> Embeddings:
    """按配置返回 Embedding；维度变化须配合向量版本迁移（PHASE_4 §3.4）。"""
    p = (provider or PROVIDER).strip().lower()
    if p not in pconfig.EMB_CONFIG:
        raise UnknownProviderError(
            f"Provider {p!r} 无 Embedding 实现；可选：{sorted(pconfig.EMB_CONFIG)}"
        )
    cfg = pconfig.EMB_CONFIG[p]

    if p == "fake":
        return FakeEmbeddings(dim=cfg["dim"])
    if p == "ollama":
        try:
            sdk = _require_sdk("langchain_ollama", "langchain-ollama", p)
            return sdk.OllamaEmbeddings(model=cfg["model"], base_url=cfg["base_url"])
        except MissingProviderSDKError:
            return _OllamaEmbeddingsHTTP(model=cfg["model"], base_url=cfg["base_url"])
    if p == "openai":
        sdk = _require_sdk("langchain_openai", "langchain-openai", p)
        return sdk.OpenAIEmbeddings(model=cfg["model"])
    if p == "tongyi":
        sdk = _require_sdk("langchain_openai", "langchain-openai", p)
        return sdk.OpenAIEmbeddings(
            model=cfg["model"],
            base_url=os.environ.get("HARDNESS_TONGYI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
        )
    raise UnknownProviderError(p)  # pragma: no cover


def embedding_dim() -> int:
    p = (PROVIDER or "ollama").strip().lower()
    if p not in pconfig.EMB_CONFIG:
        return 0
    return int(pconfig.EMB_CONFIG[p].get("dim") or 0)
