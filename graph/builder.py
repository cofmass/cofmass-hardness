"""StateGraph 组装与执行入口。

阶段一：顺序图 load_skills → retrieve → generate → remember，
用 langgraph 编译并挂接 saver（InMemorySaver 或长生命周期 PostgresSaver）。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import storage as store
from config import RAG_TOP_K
from graph import nodes
from graph.state import AgentState
from providers.factory import get_chat_model, get_embeddings
from storage.checkpointer import make_saver


def _msg_to_dict(m) -> dict:
    if isinstance(m, dict):
        return {"type": m.get("type") or m.get("role") or "human", "content": m.get("content", "")}
    return {"type": getattr(m, "type", "human"), "content": getattr(m, "content", "")}


class Agent:
    """应用级单例：持有 conn / saver / 编译图；请求内只 invoke。"""

    def __init__(self, conn, saver, *, llm=None, embeddings=None, storage_module=None):
        self.conn = conn
        self.saver = saver
        self.store = storage_module or store
        self.llm = llm or get_chat_model()
        self.embeddings = embeddings or get_embeddings()
        self.runtime = {
            "conn": conn,
            "store": self.store,
            "llm": self.llm,
            "embeddings": self.embeddings,
            "rag_top_k": RAG_TOP_K,
        }
        self.graph = self._compile()

    def _compile(self):
        node_fns = nodes.bind(self.runtime)
        g = StateGraph(AgentState)
        for name, fn in node_fns.items():
            g.add_node(name, fn)
        g.add_edge(START, "load_skills")
        g.add_edge("load_skills", "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "remember")
        g.add_edge("remember", END)
        return g.compile(checkpointer=self.saver)

    def invoke(self, question: str, user_id: str = "u-1", thread_id: str = "t-1") -> dict:
        state = {
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
            "thread_id": thread_id,
            "context": [],
            "skill_ctx": {},
            "answer": "",
            "written": {},
        }
        result = self.graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", []),
            "skill_ctx": result.get("skill_ctx", {}),
            "written": result.get("written", {}),
            "thread_id": thread_id,
            "step": len(result.get("messages", [])),
        }

    def history(self, thread_id: str) -> list[dict]:
        """time travel：从 saver 列出该 thread 的 checkpoint 步骤。"""
        # saver.list() 为 newest-first；翻转为时间顺序并过滤无消息的中间快照
        out = []
        for ckpt in reversed(list(self.saver.list({"configurable": {"thread_id": thread_id}}))):
            channels = (ckpt.checkpoint or {}).get("channel_values", {}) or {}
            msgs = channels.get("messages", []) or []
            if not msgs:
                continue
            out.append({"step": len(out) + 1, "messages": [_msg_to_dict(m) for m in msgs]})
        return out

    def close(self) -> None:
        if self.conn is not None and hasattr(self.conn, "close"):
            self.conn.close()


def build_agent(
    conn=None,
    saver=None,
    *,
    llm=None,
    embeddings=None,
    storage_module=None,
    close_conn: bool | None = None,
) -> Agent:
    """构造 Agent。conn / saver 均未传时按 backend 自建并可自动关闭。"""
    sm = storage_module or store
    if conn is None:
        conn = sm.connect()
        sm.setup(conn)
        if close_conn is None:
            close_conn = True
    else:
        if close_conn is None:
            close_conn = False
    if saver is None:
        saver, _ = make_saver()
    agent = Agent(conn, saver, llm=llm, embeddings=embeddings, storage_module=sm)
    if not close_conn:
        # 外部连接（API lifespan / 测试）由调用方关闭
        agent.close = lambda: None
    return agent


def run(
    question: str,
    user_id: str = "u-1",
    thread_id: str = "t-1",
    *,
    agent: Agent | None = None,
    conn=None,
    saver=None,
    llm=None,
    embeddings=None,
) -> dict:
    """一次性执行（CLI / 测试便捷入口）；服务用 Agent 单例。"""
    if agent is not None:
        return agent.invoke(question, user_id=user_id, thread_id=thread_id)
    built = build_agent(conn=conn, saver=saver, llm=llm, embeddings=embeddings)
    try:
        return built.invoke(question, user_id=user_id, thread_id=thread_id)
    finally:
        built.close()
