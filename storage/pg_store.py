"""阶段一 PostgreSQL 存储封装。

mem_store / docs / 检查点三载体物理隔离；本模块是 storage 抽象的 PG 实现。
图与节点禁止直接 import psycopg / pgvector —— 只经 storage/__init__.py 门面。
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from psycopg import Connection
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from config import EMB_DIM, EMB_MODEL, PG_DSN, RAG_TOP_K


def connect(dsn: str | None = None) -> Connection:
    """建立长生命周期连接（autocommit + dict_row + vector 编解码注册）。

    全新数据库可能尚未创建 vector 扩展；此时先建扩展再注册，避免
    'vector type not found' 导致连不上库（setup 无法执行）的死锁。
    """
    conn = Connection.connect(dsn or PG_DSN, autocommit=True, row_factory=dict_row)
    try:
        register_vector(conn)
    except Exception:
        # vector 类型不存在：建扩展后重试注册（幂等，已存在则 IF NOT EXISTS 无副作用）
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
    return conn


def setup(conn: Connection) -> None:
    """幂等建表：执行 db/schema.sql（CREATE EXTENSION / IF NOT EXISTS）。"""
    from pathlib import Path

    schema = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    with conn.cursor() as cur:
        cur.execute(schema.read_text(encoding="utf-8"))


def extension_version(conn: Connection) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
        row = cur.fetchone()
    return row["extversion"] if row else None


# ---------------- mem_store（长期记忆 / skill 偏好）----------------

def ns(*parts: str) -> str:
    """namespace 落库为 JSON 数组字符串，杜绝 u-1 / u-10 前缀歧义。"""
    return json.dumps(list(parts), ensure_ascii=False)


def mem_put(conn: Connection, namespace: str, key: str, value: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO mem_store (namespace, key, value)
               VALUES (%s, %s, %s::jsonb)
               ON CONFLICT (namespace, key)
               DO UPDATE SET value = EXCLUDED.value, updated_at = now()""",
            (namespace, key, json.dumps(value, ensure_ascii=False)),
        )


def mem_get(conn: Connection, namespace: str, key: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT value FROM mem_store WHERE namespace=%s AND key=%s",
            (namespace, key),
        )
        row = cur.fetchone()
    return row["value"] if row else None


def mem_search(conn: Connection, namespace: str) -> dict[str, dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT key, value FROM mem_store WHERE namespace=%s ORDER BY key",
            (namespace,),
        )
        rows = cur.fetchall()
    return {r["key"]: r["value"] for r in rows}


def mem_delete(conn: Connection, namespace: str, key: str | None = None) -> int:
    with conn.cursor() as cur:
        if key is None:
            cur.execute("DELETE FROM mem_store WHERE namespace=%s", (namespace,))
        else:
            cur.execute(
                "DELETE FROM mem_store WHERE namespace=%s AND key=%s",
                (namespace, key),
            )
        return cur.rowcount


# ---------------- docs（RAG 向量）----------------

def _to_pgvector(vec: Iterable[float]):
    from pgvector import Vector

    values = [float(x) for x in vec]
    if len(values) != EMB_DIM:
        raise ValueError(f"向量维度不一致: {len(values)}，期望 {EMB_DIM}")
    return Vector(values)


def doc_add(
    conn: Connection,
    content: str,
    embedding: Iterable[float],
    embedding_model: str = EMB_MODEL,
    metadata: dict | None = None,
    source: str | None = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO docs (content, metadata, embedding, embedding_model, source)
               VALUES (%s, %s::jsonb, %s, %s, %s) RETURNING id""",
            (
                content,
                json.dumps(metadata or {}, ensure_ascii=False),
                _to_pgvector(embedding),
                embedding_model,
                source,
            ),
        )
        return int(cur.fetchone()["id"])


def doc_search(
    conn: Connection,
    query_vec: Iterable[float],
    top_k: int = RAG_TOP_K,
    metadata_filter: dict | None = None,
    embedding_model: str | None = EMB_MODEL,
) -> list[dict]:
    qvec = _to_pgvector(query_vec)
    sql = [
        "SELECT id, content, metadata, source,",
        "       1 - (embedding <=> %s::vector) AS score",
        "FROM docs WHERE 1=1",
    ]
    params: list[Any] = [qvec]
    if metadata_filter:
        sql.append("AND metadata @> %s::jsonb")
        params.append(json.dumps(metadata_filter, ensure_ascii=False))
    if embedding_model:
        sql.append("AND embedding_model = %s")
        params.append(embedding_model)
    sql.extend(["ORDER BY embedding <=> %s::vector", "LIMIT %s"])
    params.extend([qvec, top_k])
    with conn.cursor() as cur:
        cur.execute("\n".join(sql), params)
        rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "content": r["content"],
            "metadata": r["metadata"],
            "score": float(r["score"]),
            "source": r["source"],
        }
        for r in rows
    ]
