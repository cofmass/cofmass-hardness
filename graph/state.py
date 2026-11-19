"""图状态 schema（阶段一 + 阶段三 tool 字段）。

用 TypedDict 作为 StateGraph schema：每个键声明 reducer / 默认值语义，
避免裸 dict 架构下未声明键在节点间被丢弃。
阶段三启用预留的 tool_calls / tool_results（messages 用 add_messages reducer），
新增 tool_iterations（循环收敛计数）与 memory_candidates（自动抽取候选）。
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
    tenant_id: Annotated[str | None, _replace]  # 阶段二 ACL 上下文
    thread_id: Annotated[str, _replace]
    context: Annotated[list[dict], _replace]
    skill_ctx: Annotated[dict, _replace]
    answer: Annotated[str, _replace]
    written: Annotated[dict, _replace]
    # ---- 阶段三 ----
    tool_calls: Annotated[list[dict], _replace]
    tool_results: Annotated[list[dict], _replace]
    tool_iterations: Annotated[int, _replace]
    memory_candidates: Annotated[list[dict], _replace]
    budget_meta: Annotated[dict, _replace]  # 阶段四 token 预算裁剪摘要


def get_text_input(state: dict) -> str:
    """从 state.messages 取最后一条 human 文本。"""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "type", None) == "human":
            c = getattr(m, "content", "")
            return c if isinstance(c, str) else str(c)
    return ""
