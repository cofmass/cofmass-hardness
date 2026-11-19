"""阶段三工具层：工具定义 + 注册表（ACL 由闭包强注入）。

红线（PHASE_3 §3.1 / §3.4）：
- 暴露给 LLM 的 schema 只含业务参数（question / expression），**不含 user_id/tenant_id/runtime**；
- ACL/runtime 由 `bind(runtime, user_id, tenant_id)` 闭包注入，模型无法伪造；
- 工具内访问 docs 必须复用阶段二 store.doc_search 的 metadata_filter，不越权；
- ToolNode 使用同一组绑定工具，因此执行路径与 schema 路径 ACL 一致。

示例工具：knowledge_query（知识检索）、calculator（纯计算）。
"""
from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from langchain_core.tools import StructuredTool

from config import EMB_MODEL, RAG_TOP_K

# 计算器支持的运算（不用 eval）。
_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
}


def _calc_eval(node, expression: str) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _calc_eval(node.left, expression), _calc_eval(node.right, expression)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("除数不能为 0")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _calc_eval(node.operand, expression)
        return val if isinstance(node.op, ast.UAdd) else -val
    raise ValueError("表达式含不支持的语法")


def _calculator_impl(expression: str) -> str:
    try:
        value = _calc_eval(ast.parse(expression.strip(), mode="eval").body, expression)
    except Exception as exc:
        return f"计算失败：{exc}"
    return str(int(value)) if value.is_integer() else str(value)


def _knowledge_query_impl(question: str, *, runtime: dict, user_id: str,
                          tenant_id: str | None) -> str:
    store, conn, emb = runtime.get("store"), runtime.get("conn"), runtime.get("embeddings")
    if store is None or conn is None or emb is None:
        return "知识库未就绪"
    metadata_filter: dict[str, str] = {"user_id": user_id}
    if tenant_id:
        metadata_filter["tenant_id"] = tenant_id
    kwargs: dict[str, Any] = {
        "top_k": runtime.get("rag_top_k", RAG_TOP_K),
        "metadata_filter": metadata_filter,
        "embedding_model": EMB_MODEL,
    }
    if getattr(store, "__name__", "") != "storage.sqlite_store":
        kwargs["query_text"] = question
    hits = store.doc_search(conn, emb.embed_query(question), **kwargs)
    if not hits:
        return "未检索到相关资料"
    return "\n".join(f"[{i + 1}] {h.get('content', '')}" for i, h in enumerate(hits))


def _make_tools(runtime: dict | None, user_id: str | None,
                tenant_id: str | None) -> list[StructuredTool]:
    """构造工具；ACL 与 runtime 由闭包捕获，不出现在 args_schema。"""
    rt = runtime or {}
    uid = user_id or "anonymous"

    def knowledge_query(question: str) -> str:
        """按当前用户的知识库检索资料；question 为检索问题，返回命中的文档片段。"""
        return _knowledge_query_impl(question, runtime=rt, user_id=uid, tenant_id=tenant_id)

    def calculator(expression: str) -> str:
        """计算简单算术表达式（支持 + - * / 与括号、数字），返回数值结果。"""
        return _calculator_impl(expression)

    return [
        StructuredTool.from_function(knowledge_query, name="knowledge_query"),
        StructuredTool.from_function(calculator, name="calculator"),
    ]


# 未绑定的基础工具：仅用于 schema 展示/描述；实际执行一律走 bind() 产生的闭包工具。
TOOLS = _make_tools(None, "anonymous", None)
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def bind(runtime: dict, user_id: str, tenant_id: str | None = None) -> list[StructuredTool]:
    """返回 ACL 强注入的工具列表（图与 ToolNode 都使用本函数的返回值）。"""
    return _make_tools(runtime, user_id, tenant_id)


def get_tools() -> list[StructuredTool]:
    return list(TOOLS)


def invoke_tool(name: str, args: dict, *, bound_tools: list[StructuredTool]) -> str:
    """测试/降级路径直接调用已绑定工具。"""
    t = next((x for x in bound_tools if x.name == name), None)
    if t is None:
        return f"未知工具：{name}"
    result = t.invoke(args)
    return result if isinstance(result, str) else str(result)
