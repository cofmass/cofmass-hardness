"""Provider 配置层：把 factory 从具体 SDK 中解耦。

所有值只读一次（import 时）；测试可用 monkeypatch 覆盖 get_chat_model /
get_embeddings，图与存储代码不感知这里的实现细节。
"""
from __future__ import annotations

import os

from config import CHAT_MODEL, EMB_DIM, EMB_MODEL, OLLAMA_BASE_URL, PROVIDER

# 可选 Provider 的额外配置（默认值走各家官方 SDK 的环境变量约定）。
OPENAI_CHAT_MODEL = os.getenv("HARDNESS_OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMB_MODEL = os.getenv("HARDNESS_OPENAI_EMB_MODEL", "text-embedding-3-small")
TONGYI_CHAT_MODEL = os.getenv("HARDNESS_TONGYI_CHAT_MODEL", "qwen-plus")
TONGYI_EMB_MODEL = os.getenv("HARDNESS_TONGYI_EMB_MODEL", "text-embedding-v3")
CLAUDE_CHAT_MODEL = os.getenv("HARDNESS_CLAUDE_CHAT_MODEL", "claude-3-5-sonnet-latest")

# 维度校验：切换 Embedding 模型时，若维度与当前 docs 列不匹配，拒绝静默写入。
STRICT_EMB_DIM = os.getenv("HARDNESS_STRICT_EMB_DIM", "true").lower() not in ("0", "false", "no", "off")

CHAT_CONFIG: dict[str, dict] = {
    "ollama": {"model": CHAT_MODEL, "base_url": OLLAMA_BASE_URL, "temperature": 0.0},
    "fake": {"model": "fake-chat"},
    "openai": {"model": OPENAI_CHAT_MODEL, "temperature": 0.0},
    "tongyi": {"model": TONGYI_CHAT_MODEL, "temperature": 0.0},
    "claude": {"model": CLAUDE_CHAT_MODEL, "temperature": 0.0},
}

EMB_CONFIG: dict[str, dict] = {
    "ollama": {"model": EMB_MODEL, "base_url": OLLAMA_BASE_URL, "dim": EMB_DIM},
    "fake": {"model": "fake-embedding", "dim": EMB_DIM},
    "openai": {"model": OPENAI_EMB_MODEL, "dim": EMB_DIM},
    "tongyi": {"model": TONGYI_EMB_MODEL, "dim": EMB_DIM},
}

SUPPORTED_PROVIDERS = frozenset(CHAT_CONFIG) | {"claude"}  # claude 仅 chat，embedding 需另配
