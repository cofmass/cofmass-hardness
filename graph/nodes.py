"""对话图节点：load_skills → retrieve → generate → remember。

阶段二：
- `load_skills` / `remember` 优先走注入的 PGStore(BaseStore)，sql 回退仍用阶段一 SQL 封装；
- `retrieve` 传 ACL metadata 过滤（user_id + tenant_id）与 query_text 以启用混合检索；
- 节点只依赖 providers.* / storage.* 抽象，不 import 具体 SDK。
阶段三：
- `generate` 支持 bind_tools(TOOLS) 与循环上限；
- `extract_memory` 在 remember 前运行（默认关闭）。
namespace 约定与阶段四 RLS 对齐：(tenant_id?, user_id, <area>)。
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from config import EMB_MODEL, TOOL_MAX_ITERATIONS, TOOLS_ENABLED
from graph.state import AgentState, get_text_input as _input_text


def _acl_ns(state: dict, area: str) -> tuple[str, ...]:
    """阶段二 ACL namespace：末尾为 area；user/tenant 由 state 提供。"""
    user_id = state.get("user_id", "anonymous")
    tenant_id = state.get("tenant_id")
    head = (tenant_id, user_id) if tenant_id else (user_id,)
    return head + (area,)


def _store_get(rt: dict, namespace: tuple[str, ...], key: str):
    """优先注入的 PGStore(BaseStore).get；否则回退阶段一 SQL 封装。"""
    lg_store = rt.get("langgraph_store")
    if lg_store is not None:
        item = lg_store.get(namespace, key)
        return item.value if item is not None else None
    sql_ns = json.dumps(list(namespace), ensure_ascii=False)
    return rt["store"].mem_get(rt["conn"], sql_ns, key)


def _store_put(rt: dict, namespace: tuple[str, ...], key: str, value: dict) -> None:
    lg_store = rt.get("langgraph_store")
    if lg_store is not None:
        lg_store.put(namespace, key, value)
        return
    sql_ns = json.dumps(list(namespace), ensure_ascii=False)
    rt["store"].mem_put(rt["conn"], sql_ns, key, value)


def load_skills(state: dict, *, runtime: dict | None = None) -> dict:
    rt = runtime or {}
    skills_ns = _acl_ns(state, "skills")
    prefs = _store_get(rt, skills_ns, "preferences") or {}
    enabled = _store_get(rt, skills_ns, "enabled") or {}
    return {
        "skill_ctx": {
            "language": prefs.get("language", "zh"),
            "verbosity": prefs.get("verbosity", "concise"),
            "auto_invoke": prefs.get("auto_invoke", False),
            "enabled_skills": enabled.get("list", []),
        }
    }


def retrieve(state: dict, *, runtime: dict | None = None) -> dict:
    rt = runtime or {}
    store, conn, emb = rt["store"], rt["conn"], rt["embeddings"]
    question = _input_text(state)
    if not question.strip():
        return {"context": []}
    qvec = emb.embed_query(question)
    metadata_filter = {"user_id": state.get("user_id", "anonymous")}
    if state.get("tenant_id"):
        metadata_filter["tenant_id"] = state["tenant_id"]
    search_kwargs = dict(
        top_k=rt.get("rag_top_k", 3),
        metadata_filter=metadata_filter,
        embedding_model=EMB_MODEL,
    )
    # 混合检索 query_text 仅门面/PG 后端支持；直接注入 sqlite_store 的测试保持兼容。
    if getattr(store, "__name__", "") != "storage.sqlite_store":
        search_kwargs["query_text"] = question
    hits = store.doc_search(conn, qvec, **search_kwargs)
    return {"context": hits}


def generate(state: dict, *, runtime: dict | None = None) -> dict:
    rt = runtime or {}
    llm = rt["llm"]
    skill = state.get("skill_ctx") or {}
    parts = ["你是一个有帮助的助手。"]
    if skill.get("language") == "zh":
        parts.append("请用中文回答。")
    if skill.get("verbosity") == "concise":
        parts.append("回答简洁。")
    if skill.get("enabled_skills"):
        parts.append("可用技能：" + "、".join(skill["enabled_skills"]))

    context = state.get("context") or []
    if context:
        snippets = "\n\n".join(f"[资料{i+1}] {h['content']}" for i, h in enumerate(context))
        parts.append("以下是参考资料，回答时优先依据：\n" + snippets)

    messages = [SystemMessage(content="\n".join(parts))] + list(state.get("messages", []))
    iterations = int(state.get("tool_iterations") or 0)
    tools_on = bool(rt.get("tools_enabled", TOOLS_ENABLED))
    # 循环超限：降级为无工具回答，避免死循环（DoD：收敛且不崩）。
    if tools_on and iterations >= TOOL_MAX_ITERATIONS:
        parts.append("工具调用次数已达上限，请直接基于已有信息回答，不要再调用工具。")
        tools_on = False
        messages = [SystemMessage(content="\n".join(parts))] + list(state.get("messages", []))

    call_llm = llm
    if tools_on:
        # 用当前请求的 ACL 绑定工具（events/builder 中同一实例），schema 不含 ACL 字段。
        from tools.registry import bind as bind_tools_with_acl
        bound = bind_tools_with_acl(rt, state.get("user_id", "anonymous"), state.get("tenant_id"))
        try:
            call_llm = llm.bind_tools(bound)
        except Exception:
            # Provider 不支持 bind_tools：降级无工具路径（PHASE_3 §6 风险应对）。
            call_llm = llm
            tools_on = False

    resp = call_llm.invoke(messages)
    answer = getattr(resp, "content", str(resp)) or ""
    tool_calls = list(getattr(resp, "tool_calls", None) or [])

    out: dict = {
        "answer": answer,
        "tool_calls": tool_calls,
        "messages": list(state.get("messages", [])) + [resp],
    }
    if tool_calls:
        out["tool_iterations"] = iterations + 1
        out["tool_results"] = list(state.get("tool_results") or []) + [{
            "iteration": iterations + 1,
            "calls": [{"name": c.get("name", ""), "args": c.get("args", {})} for c in tool_calls],
        }]
    if not answer and tool_calls:
        # 工具调用轮次通常无正文：answer 保持空，SSE 不推送 answer 事件。
        out.pop("answer", None)
        out["answer"] = ""
    return out


def remember(state: dict, *, runtime: dict | None = None) -> dict:
    """显式写入：仅处理 '/remember key=value'，不做自动抽取。"""
    rt = runtime or {}
    user_id = state.get("user_id", "anonymous")
    text = _input_text(state)
    written: dict = {}
    if text.startswith("/remember "):
        body = text[len("/remember "):].strip()
        if "=" in body:
            key, raw = body.split("=", 1)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"text": raw}
            if not isinstance(parsed, dict):
                parsed = {"text": raw}
            key = key.strip()
            _store_put(rt, _acl_ns(state, "facts"), key, parsed)
            written[key] = parsed
    return {"written": written}


def set_skills(
    user_id: str,
    store,
    conn,
    preferences: dict | None = None,
    enabled: list[str] | None = None,
    tenant_id: str | None = None,
    langgraph_store=None,
) -> None:
    """显式设置用户 skill 偏好（PUT /skills 走同一 UPSERT）。

    `store` 必须是带 mem_put/mem_get 的 SQL 门面（回退路径用）；
    `langgraph_store` 为可选的 BaseStore（PGStore），优先走它的 put/get。
    两者不能混传为同一对象（G1 失败：PGStore 无 mem_put 导致 500）。
    """
    namespace = ((tenant_id, user_id) if tenant_id else (user_id,)) + ("skills",)
    rt = {"store": store, "conn": conn, "langgraph_store": langgraph_store}
    if preferences is not None:
        _store_put(rt, namespace, "preferences", preferences)
    if enabled is not None:
        _store_put(rt, namespace, "enabled", {"list": enabled, "updated_by": "user"})


def get_skills(user_id: str, store, conn, tenant_id: str | None = None,
               langgraph_store=None) -> dict:
    namespace = ((tenant_id, user_id) if tenant_id else (user_id,)) + ("skills",)
    rt = {"store": store, "conn": conn, "langgraph_store": langgraph_store}
    return {
        "preferences": _store_get(rt, namespace, "preferences") or {},
        "enabled": _store_get(rt, namespace, "enabled") or {},
    }


# ---------------- LangGraph 节点适配（注入 runtime，不污染业务函数签名）----------------


def bind(runtime: dict) -> dict[str, callable]:
    """返回可直接 add_node 的节点函数：state -> dict。"""
    def _load_skills(state: AgentState) -> dict:
        return load_skills(state, runtime=runtime)

    def _retrieve(state: AgentState) -> dict:
        return retrieve(state, runtime=runtime)

    def _generate(state: AgentState) -> dict:
        return generate(state, runtime=runtime)

    def _extract_memory(state: AgentState) -> dict:
        from graph.memory_extractor import extract_memory_node
        return extract_memory_node(state, runtime=runtime)

    def _remember(state: AgentState) -> dict:
        return remember(state, runtime=runtime)

    return {
        "load_skills": _load_skills,
        "retrieve": _retrieve,
        "generate": _generate,
        "extract_memory": _extract_memory,
        "remember": _remember,
    }
