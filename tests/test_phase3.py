"""阶段三测试：工具分支、循环上限、ACL、白名单、置信度、SSE 事件顺序、/invoke 回归。

所有测试用 fake LLM / Embedding / InMemorySaver，不依赖 Ollama / PG；
通过环境变量开关阶段三特性，测试内显式 monkeypatch 避免互相污染。
"""
from __future__ import annotations

import asyncio
import json
import os

os.environ.setdefault("HARDNESS_STORE_BACKEND", "sql")

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import config
import storage as store
from graph import nodes
from graph.builder import build_agent
from graph.memory_extractor import extract_memory_node
from storage import sqlite_store
from storage.checkpointer import make_saver

from config import EMB_DIM


class FakeEmb:
    def embed_query(self, text): return [1.0] + [0.0] * (EMB_DIM - 1)
    def embed_documents(self, texts): return [[1.0] + [0.0] * (EMB_DIM - 1) for _ in texts]


class ToolLLM:
    """按调用次数返回：第 1 次 tool_call，之后正文；可配置无限调用。"""

    def __init__(self, *, always_tool: bool = False, tool_name: str = "calculator",
                 args: dict | None = None, final: str = "最终答案"):
        self.always_tool = always_tool
        self.tool_name = tool_name
        self.args = args or {"expression": "1+1"}
        self.final = final
        self.calls = 0

    def _response(self, messages):
        self.calls += 1
        if self.always_tool or self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": self.args,
                             "id": f"call-{self.calls}"}],
            )
        return AIMessage(content=self.final)

    def invoke(self, messages):
        return self._response(messages)

    def bind_tools(self, tools):
        llm = self

        class Bound:
            def invoke(self, messages):
                return llm._response(messages)

        return Bound()

    def bind(self, *args, **kwargs):
        return self


@pytest.fixture
def conn(tmp_path):
    c = sqlite_store.connect(tmp_path / "test.sqlite3")
    sqlite_store.setup(c)
    return c


def _agent(conn, llm, *, tools=True):
    saver, _ = make_saver()
    return build_agent(conn=conn, saver=saver, llm=llm, embeddings=FakeEmb(),
                       storage_module=sqlite_store, tools_enabled=tools)


def test_tool_branch_then_final_answer(conn, monkeypatch):
    """DoD: generate 触发 ToolNode，工具结果回流 generate 生成最终回答。"""
    monkeypatch.setattr(config, "TOOL_MAX_ITERATIONS", 5)
    llm = ToolLLM(tool_name="calculator", args={"expression": "2*(3+4)"}, final="答案是 14")
    agent = _agent(conn, llm, tools=True)
    r = agent.invoke("算一下 2*(3+4)", user_id="u-1", thread_id="t-tool")
    assert r["answer"] == "答案是 14"
    assert llm.calls == 2, "应触发一次工具后再生成最终回答"
    tr = r["tool_results"]
    assert any(x.get("name") == "calculator" and x.get("status") == "success" for x in tr)
    assert any("14" in str(x.get("result")) for x in tr)


def test_tool_loop_cap_converges(conn, monkeypatch):
    """DoD: 循环超限（默认 5）收敛并记录，不死循环。"""
    monkeypatch.setattr(config, "TOOL_MAX_ITERATIONS", 3)
    llm = ToolLLM(always_tool=True)
    agent = _agent(conn, llm, tools=True)
    r = agent.invoke("一直算", user_id="u-1", thread_id="t-loop")
    # 3 次工具 + 1 次超限降级回答
    assert llm.calls <= 4
    assert len(r["tool_results"]) == 3
    assert r["answer"] in ("最终答案", ""), "降级后应给出无工具回答"


def test_tool_acl_isolation(conn):
    """DoD: 工具内数据访问带 ACL 过滤，u-1 读不到 u-2 的文档。"""
    from config import EMB_MODEL
    vec = [1.0] + [0.0] * (EMB_DIM - 1)
    sqlite_store.doc_add(conn, "秘密资料U2", vec, EMB_MODEL, {"user_id": "u-2"})
    sqlite_store.doc_add(conn, "公开资料U1", vec, EMB_MODEL, {"user_id": "u-1"})
    rt = {"store": sqlite_store, "conn": conn, "embeddings": FakeEmb(), "rag_top_k": 10}
    from tools.registry import bind, invoke_tool
    for uid, expect, forbidden in (("u-1", "公开资料U1", "秘密资料U2"),
                                   ("u-2", "秘密资料U2", "公开资料U1")):
        tools = bind(rt, uid, None)
        result = invoke_tool("knowledge_query", {"question": "资料"}, bound_tools=tools)
        assert expect in result
        assert forbidden not in result


def test_tool_schema_has_no_acl_fields():
    """ACL 字段不得出现在 LLM 可见 schema 中。"""
    from tools.registry import bind
    tools = bind({"store": None, "conn": None, "embeddings": None}, "u-1", "t-x")
    kq = next(t for t in tools if t.name == "knowledge_query")
    props = kq.args
    assert "user_id" not in props and "tenant_id" not in props and "runtime" not in props
    assert "question" in props


def _state(user_id="u-1", tenant_id=None):
    return {"messages": [HumanMessage(content="记住：我喜欢简洁的中文回复")],
            "user_id": user_id, "tenant_id": tenant_id, "thread_id": "t-mem",
            "context": [], "skill_ctx": {}, "answer": "ok", "written": {}}


class LLMCandidates:
    def __init__(self, candidate):
        self.candidate = candidate
    def invoke(self, messages):
        return AIMessage(content=json.dumps([self.candidate], ensure_ascii=False))


def test_auto_memory_disabled_by_default(conn, monkeypatch):
    """DoD: 自动抽取默认关闭。"""
    monkeypatch.setattr(config, "AUTO_MEMORY_ENABLED", False)
    rt = {"langgraph_store": None, "store": sqlite_store, "conn": conn,
          "llm": LLMCandidates({"area": "skills", "key": "preferences", "field": "language",
                                "value": "zh", "confidence": 0.99})}
    out = extract_memory_node(_state(), runtime=rt)
    assert out["memory_candidates"] == []


def test_auto_memory_whitelist_and_confidence(conn, monkeypatch):
    """DoD: 仅白名单 namespace/key 被写入，低置信度丢弃，带 updated_by=auto。"""
    monkeypatch.setattr(config, "AUTO_MEMORY_ENABLED", True)
    import graph.memory_extractor as me
    monkeypatch.setattr(me, "AUTO_MEMORY_ENABLED", True)
    from langgraph.store.memory import InMemoryStore
    bs = InMemoryStore()
    # 白名单外 area -> 丢弃
    out = extract_memory_node(_state(), runtime={
        "langgraph_store": bs, "store": sqlite_store, "conn": conn,
        "llm": LLMCandidates({"area": "secret", "key": "token", "field": "k", "value": "v", "confidence": 0.99}),
    })
    assert out["memory_candidates"] == []
    # 白名单内但置信度低 -> 丢弃
    out = extract_memory_node(_state(), runtime={
        "langgraph_store": bs, "store": sqlite_store, "conn": conn,
        "llm": LLMCandidates({"area": "skills", "key": "preferences", "field": "language",
                              "value": "zh", "confidence": 0.3}),
    })
    assert out["memory_candidates"] == []
    # 白名单内高置信度 -> 写入，带 updated_by=auto
    out = extract_memory_node(_state(), runtime={
        "langgraph_store": bs, "store": sqlite_store, "conn": conn,
        "llm": LLMCandidates({"area": "skills", "key": "preferences", "field": "language",
                              "value": "zh", "confidence": 0.95}),
    })
    assert len(out["memory_candidates"]) == 1
    item = bs.get(("u-1", "skills"), "preferences")
    assert item.value["language"] == "zh"
    assert item.value["updated_by"] == "auto"
    assert item.value["confidence"] == 0.95


def test_auto_memory_tenant_namespace(conn, monkeypatch):
    """DoD: namespace 首位由 ACL 注入，tenant 场景 (t-A, u-1, skills)。"""
    import graph.memory_extractor as me
    monkeypatch.setattr(me, "AUTO_MEMORY_ENABLED", True)
    from langgraph.store.memory import InMemoryStore
    bs = InMemoryStore()
    extract_memory_node(_state(tenant_id="t-A"), runtime={
        "langgraph_store": bs, "store": sqlite_store, "conn": conn,
        "llm": LLMCandidates({"area": "skills", "key": "preferences", "field": "language",
                              "value": "zh", "confidence": 0.95}),
    })
    assert bs.get(("t-A", "u-1", "skills"), "preferences").value["language"] == "zh"


def test_auto_memory_explicit_put_wins(conn, monkeypatch):
    """DoD: 显式 PUT 覆盖 auto（updated_by != auto 时自动写入跳过）。"""
    import graph.memory_extractor as me
    monkeypatch.setattr(me, "AUTO_MEMORY_ENABLED", True)
    from langgraph.store.memory import InMemoryStore
    bs = InMemoryStore()
    bs.put(("u-1", "skills"), "preferences",
           {"language": "en", "updated_by": "explicit"})
    extract_memory_node(_state(), runtime={
        "langgraph_store": bs, "store": sqlite_store, "conn": conn,
        "llm": LLMCandidates({"area": "skills", "key": "preferences", "field": "language",
                              "value": "zh", "confidence": 0.99}),
    })
    assert bs.get(("u-1", "skills"), "preferences").value["language"] == "en"


def test_async_stream_event_order(conn, monkeypatch):
    """DoD: SSE 事件顺序 node(start/end) → token* → answer/done，且可解析。"""
    from graph.async_builder import AsyncAgent
    monkeypatch.setattr(config, "ASYNC_ENABLED", False)  # 强制 to_thread 回退路径
    llm = ToolLLM(final="你好")
    agent = _agent(conn, llm, tools=False)
    aa = AsyncAgent(agent)
    events = asyncio.run(_collect(aa.stream_events("hi", user_id="u-1", thread_id="t-sse")))
    kinds = [e["event"] for e in events]
    assert "answer" in kinds and "done" in kinds
    assert kinds.index("answer") < kinds.index("done")
    answer = next(e for e in events if e["event"] == "answer")
    assert answer["payload"]["answer"] == "你好"
    assert answer["payload"]["degraded"] is True


def test_sse_http_endpoint(monkeypatch):
    """DoD: POST /chat/stream 返回 text/event-stream，事件按序，前端可解析。"""
    from fastapi.testclient import TestClient
    import providers.factory as pf
    monkeypatch.setattr(config, "ASYNC_ENABLED", False)
    monkeypatch.setenv("HARDNESS_STORE_BACKEND", "sql")
    monkeypatch.setattr(pf, "get_chat_model", lambda: ToolLLM(final="你好"))
    monkeypatch.setattr(pf, "get_embeddings", lambda: FakeEmb())
    import app as app_module
    client = TestClient(app_module.app)
    with client:
        resp = client.post("/api/v1/chat/stream", json={
            "user_id": "u-1", "thread_id": "t-http", "question": "你好",
        })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    kinds = [e[0] for e in events]
    assert "answer" in kinds and "done" in kinds
    assert kinds.index("answer") < kinds.index("done")


def test_invoke_regression(conn):
    """DoD: 同步 /chat/invoke 不受阶段三影响。"""
    agent = _agent(conn, ToolLLM(final="ok"), tools=False)
    r = agent.invoke("hi", user_id="u-2", thread_id="t-invoke")
    assert r["answer"] == "ok"
    assert r["thread_id"] == "t-invoke"


async def _collect(aiter):
    return [item async for item in aiter]


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = block.strip().splitlines()
        if not lines or lines[0].startswith(":"):
            continue
        event, data = None, "{}"
        for line in lines:
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if event:
            out.append((event, json.loads(data)))
    return out
