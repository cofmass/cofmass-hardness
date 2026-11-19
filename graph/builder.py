"""StateGraph 组装与执行入口。

阶段一：顺序图 load_skills → retrieve → generate → remember；
阶段三：TOOLS_ENABLED=true 时加 ToolNode + tools_condition 条件边；
默认关闭，保留顺序图回归路径。编译挂接 saver（InMemorySaver / PostgresSaver）。
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import storage as store
from config import RAG_TOP_K, TOOLS_ENABLED
from graph import nodes
from graph.state import AgentState
from providers.factory import get_chat_model, get_embeddings
from storage.checkpointer import make_saver
from tools.registry import bind as bind_tools_with_acl

log = logging.getLogger("hardness.graph")


def _msg_to_dict(m) -> dict:
    if isinstance(m, dict):
        return {"type": m.get("type") or m.get("role") or "human", "content": m.get("content", "")}
    return {"type": getattr(m, "type", "human"), "content": getattr(m, "content", "")}


class Agent:
    """应用级单例：持有 conn / saver / 编译图；请求内只 invoke。"""

    def __init__(self, conn, saver, *, llm=None, embeddings=None, storage_module=None,
                 langgraph_store=None, tools_enabled: bool | None = None):
        self.conn = conn
        self.saver = saver
        self.store = storage_module or store
        self.langgraph_store = langgraph_store if langgraph_store is not None else store.make_langgraph_store(conn)
        self.llm = llm or get_chat_model()
        self.embeddings = embeddings or get_embeddings()
        self.tools_enabled = TOOLS_ENABLED if tools_enabled is None else tools_enabled
        self.runtime = {
            "conn": conn,
            "store": self.store,  # 阶段一 SQL 门面：docs 检索 / 回退 mem_* 都用
            "langgraph_store": self.langgraph_store,  # 阶段二 PGStore(BaseStore)：mem 读写
            "llm": self.llm,
            "embeddings": self.embeddings,
            "rag_top_k": RAG_TOP_K,
            "tools_enabled": self.tools_enabled,
        }
        self._graphs = {}
        self.graph = self._compile()

    def _compile_graph(self, tools):
        """按传入工具集编译图；不同 ACL 用不同编译图实例（进程内薄缓存）。"""
        node_fns = nodes.bind(self.runtime)
        g = StateGraph(AgentState)
        for name, fn in node_fns.items():
            g.add_node(name, fn)
        g.add_edge(START, "load_skills")
        g.add_edge("load_skills", "retrieve")
        g.add_edge("retrieve", "generate")
        if self.tools_enabled:
            g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
            g.add_conditional_edges("generate", tools_condition, {
                "tools": "tools",
                "__end__": "extract_memory",
            })
            g.add_edge("tools", "generate")
        else:
            g.add_edge("generate", "extract_memory")
        g.add_edge("extract_memory", "remember")
        g.add_edge("remember", END)
        return g.compile(checkpointer=self.saver, store=self.langgraph_store)

    def _compile(self):
        from tools.registry import bind as bind_tools_with_acl
        if not self.tools_enabled:
            return self._compile_graph([])
        # 默认图使用匿名 ACL；invoke() 会按 (user_id, tenant_id) 取对应 ACL 绑定图。
        tools = bind_tools_with_acl(self.runtime, "anonymous", None)
        default = self._compile_graph(tools)
        self._graphs[(None, None)] = default
        return default

    def _graph_for(self, user_id: str | None, tenant_id: str | None):
        if not getattr(self, "tools_enabled", False):
            return self.graph
        from tools.registry import bind as bind_tools_with_acl
        key = (user_id or "anonymous", tenant_id)
        cached = getattr(self, "_graphs", {}).get(key)
        if cached is None:
            tools = bind_tools_with_acl(self.runtime, user_id or "anonymous", tenant_id)
            cached = self._graphs[key] = self._compile_graph(tools)
        return cached

    def invoke(self, question: str, user_id: str = "u-1", thread_id: str = "t-1",
               tenant_id: str | None = None, config: dict | None = None) -> dict:
        state = {
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "context": [],
            "skill_ctx": {},
            "answer": "",
            "written": {},
        }
        invoke_config = {
            "configurable": {
                "thread_id": thread_id,
                # ACL/runtime 供 ToolNode 内 InjectedToolArg 注入；模型 schema 不可见。
                "user_id": user_id,
                "tenant_id": tenant_id,
                "runtime": self.runtime,
            }
        }
        if config:
            invoke_config.update(config)
        graph = self._graph_for(user_id, tenant_id)
        result = graph.invoke(state, config=invoke_config)
        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", []),
            "skill_ctx": result.get("skill_ctx", {}),
            "written": result.get("written", {}),
            "thread_id": thread_id,
            "step": len(result.get("messages", [])),
            "tool_results": result.get("tool_results", []),
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
    langgraph_store=None,
    close_conn: bool | None = None,
    tools_enabled: bool | None = None,
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
    agent = Agent(conn, saver, llm=llm, embeddings=embeddings,
                  storage_module=sm, langgraph_store=langgraph_store,
                  tools_enabled=tools_enabled)
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
