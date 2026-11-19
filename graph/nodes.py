"""对话图节点：load_skills → retrieve → generate → remember。

节点只依赖 providers.* / storage.* 抽象；禁止直接引用具体 SDK 或传输模块。
依赖通过 runtime 注入（见 graph/builder.py），无模块级可变连接。
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, SystemMessage

from config import EMB_MODEL
from graph.state import AgentState, get_text_input as _input_text


def load_skills(state: dict, *, runtime: dict | None = None) -> dict:
    rt = runtime or {}
    store, conn = rt["store"], rt["conn"]
    user_id = state.get("user_id", "anonymous")
    skills_ns = store.ns(user_id, "skills")
    prefs = store.mem_get(conn, skills_ns, "preferences") or {}
    enabled = store.mem_get(conn, skills_ns, "enabled") or {}
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
    hits = store.doc_search(
        conn,
        qvec,
        top_k=rt.get("rag_top_k", 3),
        metadata_filter={"user_id": state.get("user_id", "anonymous")},
        embedding_model=EMB_MODEL,
    )
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
    resp = llm.invoke(messages)
    answer = getattr(resp, "content", str(resp))
    return {
        "answer": answer,
        "messages": list(state.get("messages", [])) + [AIMessage(content=answer)],
    }


def remember(state: dict, *, runtime: dict | None = None) -> dict:
    """显式写入：仅处理 '/remember key=value'，不做自动抽取。"""
    rt = runtime or {}
    store, conn = rt["store"], rt["conn"]
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
            store.mem_put(conn, store.ns(user_id, "facts"), key, parsed)
            written[key] = parsed
    return {"written": written}


def set_skills(
    user_id: str,
    store,
    conn,
    preferences: dict | None = None,
    enabled: list[str] | None = None,
) -> None:
    """显式设置用户 skill 偏好（PUT /skills 走同一 UPSERT）。"""
    ns = store.ns(user_id, "skills")
    if preferences is not None:
        store.mem_put(conn, ns, "preferences", preferences)
    if enabled is not None:
        store.mem_put(conn, ns, "enabled", {"list": enabled, "updated_by": "user"})


def get_skills(user_id: str, store, conn) -> dict:
    ns = store.ns(user_id, "skills")
    return {
        "preferences": store.mem_get(conn, ns, "preferences") or {},
        "enabled": store.mem_get(conn, ns, "enabled") or {},
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

    def _remember(state: AgentState) -> dict:
        return remember(state, runtime=runtime)

    return {
        "load_skills": _load_skills,
        "retrieve": _retrieve,
        "generate": _generate,
        "remember": _remember,
    }
