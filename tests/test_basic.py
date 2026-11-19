"""阶段一测试：契约测试（fake LLM/Emb，不依赖外部）+ PG 集成测试（无 PG 明确 skip）。"""
from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessage

# 契约测试必须固定 sql 后端：否则本机 .env 配了 DSN 时全局门面会绑到 pg_store，
# 而测试 fixture 传的是 sqlite 连接，导致 PG/SQLite 混用（G1 失败 #1）。
os.environ.setdefault("HARDNESS_STORE_BACKEND", "sql")

import storage as store
from config import EMB_DIM
from graph.builder import build_agent
from graph.nodes import get_skills, set_skills
from storage import sqlite_store
from storage.checkpointer import make_saver


class FakeEmb:
    def embed_query(self, text): return [1.0] + [0.0] * (EMB_DIM - 1)
    def embed_documents(self, texts): return [[1.0] + [0.0] * (EMB_DIM - 1) for _ in texts]


class FakeLLM:
    def invoke(self, messages): return AIMessage(content="ok")


@pytest.fixture
def conn(tmp_path):
    c = sqlite_store.connect(tmp_path / "test.sqlite3")
    sqlite_store.setup(c)
    return c


@pytest.fixture
def agent(conn):
    saver, _ = make_saver()
    return build_agent(
        conn=conn, saver=saver,
        llm=FakeLLM(), embeddings=FakeEmb(),
        storage_module=sqlite_store,
    )


def test_namespace_no_prefix_collision(conn):
    ns1, ns10 = store.ns("u-1", "skills"), store.ns("u-10", "skills")
    store.mem_put(conn, ns1, "preferences", {"language": "zh"})
    store.mem_put(conn, ns10, "preferences", {"language": "en"})
    assert store.mem_get(conn, ns1, "preferences")["language"] == "zh"
    assert store.mem_get(conn, ns10, "preferences")["language"] == "en"


def test_skill_preferences_roundtrip(conn):
    set_skills("u-1", sqlite_store, conn, {"language": "zh", "verbosity": "concise"}, ["code-review"])
    ctx = get_skills("u-1", sqlite_store, conn)
    assert ctx["preferences"]["language"] == "zh"
    assert ctx["enabled"]["list"] == ["code-review"]


def test_skill_preferences_with_injected_basestore(conn):
    """回归 G1 失败 #2：API 把 PGStore(BaseStore) 注入后 skills 读写 500。

    API 实际调用是 set_skills(SQL 门面, langgraph_store=BaseStore)；
    旧实现把 BaseStore 当 SQL 门面传给 mem_put/mem_get 导致 AttributeError。
    这里用内存 BaseStore 模拟 PGStore，验证注入路径正确且跨用户隔离。
    """
    from langgraph.store.memory import InMemoryStore

    bs = InMemoryStore()
    set_skills("u-1", sqlite_store, conn, {"language": "zh", "verbosity": "concise"},
               ["code-review"], langgraph_store=bs)
    ctx = get_skills("u-1", sqlite_store, conn, langgraph_store=bs)
    assert ctx["preferences"]["language"] == "zh"
    assert ctx["enabled"]["list"] == ["code-review"]
    # 写入落在 BaseStore 的 user namespace，而非 SQL 门面
    ns1 = ("u-1", "skills")
    assert bs.get(ns1, "preferences").value["language"] == "zh"
    # 租户 namespace 前缀
    set_skills("u-1", sqlite_store, conn, {"language": "en"},
               langgraph_store=bs, tenant_id="t-A")
    assert bs.get(("t-A", "u-1", "skills"), "preferences").value["language"] == "en"
    # 跨用户隔离：u-2 读不到 u-1 的偏好
    ctx2 = get_skills("u-2", sqlite_store, conn, langgraph_store=bs)
    assert ctx2["preferences"] == {}


def test_doc_metadata_isolation(conn):
    vec = [1.0, 0.0]
    id1 = sqlite_store.doc_add(conn, "A", vec, "test-embed", {"user_id": "u-1"})
    id2 = sqlite_store.doc_add(conn, "B", vec, "test-embed", {"user_id": "u-2"})
    r1 = sqlite_store.doc_search(conn, vec, 10, {"user_id": "u-1"}, "test-embed")
    r2 = sqlite_store.doc_search(conn, vec, 10, {"user_id": "u-2"}, "test-embed")
    assert [r["id"] for r in r1] == [id1]
    assert [r["id"] for r in r2] == [id2]


def test_graph_contract_fake_llm(agent):
    r1 = agent.invoke("q1", "u-1", "t-x")
    r2 = agent.invoke("q2", "u-1", "t-x")
    assert r1["answer"] == "ok" and r2["answer"] == "ok"
    assert r1["thread_id"] == r2["thread_id"] == "t-x"


def test_graph_multi_turn_checkpoint(agent):
    agent.invoke("第一轮", "u-1", "t-multi")
    agent.invoke("第二轮", "u-1", "t-multi")
    hist = agent.history("t-multi")
    assert hist, "应有 checkpoint 落库"
    # checkpoint 按节点写入；断言最终快照累积了两轮消息（2 human + 2 ai）
    final_msgs = hist[-1]["messages"]
    assert len(final_msgs) >= 4
    assert sum(1 for m in final_msgs if m.get("type") == "human") >= 2


def test_provider_swappable_only_fake(agent):
    """换 Provider 只注入 fake，图不依赖具体 SDK。"""
    assert agent.invoke("hi", "u-2", "t-p")["answer"] == "ok"


# ---- PG 集成测试：无 PG 时明确 skip，不伪报通过 ----
def test_pg_schema_and_saver():
    from config import PG_DSN
    if not PG_DSN:
        pytest.skip("本机未配置 HARDNESS_PG_DSN，PG 集成测试跳过（不伪报通过）")
    from storage import pg_store
    conn = pg_store.connect()
    pg_store.setup(conn)
    assert pg_store.extension_version(conn), "pgvector 扩展未就绪"
    saver, _ = make_saver()
    a = build_agent(conn=conn, saver=saver, llm=FakeLLM(), embeddings=FakeEmb(), storage_module=pg_store)
    assert a.invoke("pg q", "u-1", "t-pg")["answer"] == "ok"
    conn.close()


def test_graph_no_concrete_sdk_import():
    """静态检查：graph/ 内无具体 SDK / 传输代码 import。"""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "graph"
    pattern = re.compile(r"\b(import\s+(psycopg|pgvector|urllib)|from\s+(psycopg|pgvector|urllib))")
    for f in root.glob("*.py"):
        assert not pattern.search(f.read_text(encoding="utf-8")), f"{f.name} 含具体 SDK import"
