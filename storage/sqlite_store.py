"""阶段一本地存储：SQLite 同时承载 checkpoint / mem_store / docs。

设计对齐 AGENT_ARCHITECTURE.md §5，区别仅在物理载体：
  - checkpoint_*  → checkpoints 表（自建简化版，够 time travel / 断点续跑）
  - mem_store     → namespace 存 JSON 数组字符串，避免 'u-1' 与 'u-10' 前缀歧义
  - docs          → embedding 存 BLOB（float32），检索用余弦相似度
"""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any, Iterable

from config import SQLITE_PATH


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def setup(conn: sqlite3.Connection) -> None:
    """幂等建表；服务启动时调用即可（对应 PostgresSaver.setup()）。"""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (thread_id, step)
        );

        CREATE TABLE IF NOT EXISTS mem_store (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (namespace, key)
        );

        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            embedding BLOB NOT NULL,
            embedding_model TEXT NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()


# ---------------- checkpoint（短期记忆）----------------

def save_checkpoint(conn: sqlite3.Connection, thread_id: str, step: int, state: dict) -> None:
    conn.execute(
        """INSERT INTO checkpoints (thread_id, step, state_json)
           VALUES (?, ?, ?)
           ON CONFLICT(thread_id, step) DO UPDATE SET
             state_json=excluded.state_json, updated_at=datetime('now')""",
        (thread_id, step, json.dumps(state, ensure_ascii=False, default=str)),
    )
    conn.commit()


def load_checkpoint(conn: sqlite3.Connection, thread_id: str) -> tuple[int, dict] | None:
    row = conn.execute(
        "SELECT step, state_json FROM checkpoints WHERE thread_id=? ORDER BY step DESC LIMIT 1",
        (thread_id,),
    ).fetchone()
    if not row:
        return None
    return row["step"], json.loads(row["state_json"])


def history(conn: sqlite3.Connection, thread_id: str) -> list[tuple[int, dict]]:
    rows = conn.execute(
        "SELECT step, state_json FROM checkpoints WHERE thread_id=? ORDER BY step",
        (thread_id,),
    ).fetchall()
    return [(r["step"], json.loads(r["state_json"])) for r in rows]


# ---------------- mem_store（长期记忆 / skill 偏好）----------------

def ns(*parts: str) -> str:
    """namespace 落库为 JSON 数组字符串，杜绝前缀歧义。"""
    return json.dumps(list(parts), ensure_ascii=False)


def mem_put(conn: sqlite3.Connection, namespace: str, key: str, value: dict) -> None:
    conn.execute(
        """INSERT INTO mem_store (namespace, key, value)
           VALUES (?, ?, ?)
           ON CONFLICT(namespace, key) DO UPDATE SET
             value=excluded.value, updated_at=datetime('now')""",
        (namespace, key, json.dumps(value, ensure_ascii=False)),
    )
    conn.commit()


def mem_get(conn: sqlite3.Connection, namespace: str, key: str) -> dict | None:
    row = conn.execute(
        "SELECT value FROM mem_store WHERE namespace=? AND key=?", (namespace, key)
    ).fetchone()
    return json.loads(row["value"]) if row else None


def mem_search(conn: sqlite3.Connection, namespace: str) -> dict[str, dict]:
    rows = conn.execute(
        "SELECT key, value FROM mem_store WHERE namespace=? ORDER BY key", (namespace,)
    ).fetchall()
    return {r["key"]: json.loads(r["value"]) for r in rows}


def mem_delete(conn: sqlite3.Connection, namespace: str, key: str | None = None) -> int:
    if key is None:
        cur = conn.execute("DELETE FROM mem_store WHERE namespace=?", (namespace,))
    else:
        cur = conn.execute("DELETE FROM mem_store WHERE namespace=? AND key=?", (namespace, key))
    conn.commit()
    return cur.rowcount


# ---------------- docs（RAG 向量）----------------

def _pack(vec: Iterable[float]) -> bytes:
    values = tuple(vec)
    return struct.pack("<%df" % len(values), *values)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def doc_add(
    conn: sqlite3.Connection,
    content: str,
    embedding: list[float],
    embedding_model: str,
    metadata: dict | None = None,
    source: str | None = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO docs (content, metadata, embedding, embedding_model, source)
           VALUES (?, ?, ?, ?, ?)""",
        (
            content,
            json.dumps(metadata or {}, ensure_ascii=False),
            _pack(embedding),
            embedding_model,
            source,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def doc_search(
    conn: sqlite3.Connection,
    query_vec: list[float],
    top_k: int,
    metadata_filter: dict | None = None,
    embedding_model: str | None = None,
) -> list[dict]:
    rows = conn.execute("SELECT * FROM docs").fetchall()
    results = []
    for r in rows:
        if embedding_model and r["embedding_model"] != embedding_model:
            continue
        meta = json.loads(r["metadata"])
        if metadata_filter and not all(meta.get(k) == v for k, v in metadata_filter.items()):
            continue
        results.append(
            {
                "id": r["id"],
                "content": r["content"],
                "metadata": meta,
                "score": _cosine(query_vec, _unpack(r["embedding"])),
                "source": r["source"],
            }
        )
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
