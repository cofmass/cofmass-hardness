"""图状态 schema（阶段一，不含 tool_calls —— 留阶段三）。

用 TypedDict 作为 StateGraph schema：每个键声明 reducer / 默认值语义，
避免裸 dict 架构下未声明键在节点间被丢弃。
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def _replace(_old, new):
    """覆盖式更新（skill_ctx / answer 等单值字段）。"""
    return new


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: Annotated[str, _replace]
    thread_id: Annotated[str, _replace]
    context: Annotated[list[dict], _replace]
    skill_ctx: Annotated[dict, _replace]
    answer: Annotated[str, _replace]
    written: Annotated[dict, _replace]


def get_text_input(state: dict) -> str:
    """从 state.messages 取最后一条 human 文本。"""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", None) == "human":
            c = getattr(m, "content", "")
            return c if isinstance(c, str) else str(c)
    return ""
