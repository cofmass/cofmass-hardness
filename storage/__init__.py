"""存储门面：图/服务只依赖本模块，不感知 PostgreSQL 或 SQLite。

后端由本地配置决定：
- `HARDNESS_STORE_BACKEND=pgstore`（默认随 DSN 自动推导）→ PostgreSQL + pgvector；
- `HARDNESS_STORE_BACKEND=sql` → SQLite 脚手架（本机无 PG / 显式契约测试用）。

阶段二起 `pgstore` 后端同时暴露 `PGStore(BaseStore)`；`sql` 为回退开关。
"""
from __future__ import annotations

import os

from config import PG_DSN, STORE_BACKEND, use_postgres
from storage import pg_store, sqlite_store

# 显式优先于自动推导；合法值 sql / pgstore；自动模式下由 DSN 决定。
_backend_choice = os.getenv("HARDNESS_STORE_BACKEND", "").strip().lower()
if _backend_choice not in ("", "sql", "pgstore"):
    raise ValueError(f"HARDNESS_STORE_BACKEND 只能为 sql/pgstore，收到: {_backend_choice!r}")
_use_pg = use_postgres() if not _backend_choice else (_backend_choice == "pgstore")
_backend = pg_store if _use_pg else sqlite_store


def backend_name() -> str:
    if _use_pg:
        return "postgres+pgvector(pgstore)" if _backend_choice else "postgres+pgvector"
    return "sqlite(scaffold)"


# 统一基础存储 API（两后端同签名），图/路由只经此访问。
connect = _backend.connect
setup = _backend.setup
ns = _backend.ns
mem_put = _backend.mem_put
mem_get = _backend.mem_get
mem_search = _backend.mem_search
mem_delete = _backend.mem_delete
doc_add = _backend.doc_add
doc_search = _backend.doc_search

# SQLite 脚手架专有（CLI 运维 / 旧测试用），PG 模式下为 None。
history = getattr(_backend, "history", None)


def extension_version(conn) -> str | None:
    if _use_pg:
        return pg_store.extension_version(conn)
    return "n/a"


def make_langgraph_store(conn):
    """pgstore 后端返回 PGStore(BaseStore) 供 compile(store=...)；sql 回退返回 None。"""
    if not _use_pg:
        return None
    return pg_store.PGStore(conn)


def doc_search(conn, query_vec, *args, **kwargs):
    """统一检索入口：PG 后端支持 query_text 混合检索；SQLite 忽略阶段二专有参数。"""
    if not _use_pg:
        kwargs.pop("query_text", None)
        kwargs.pop("hybrid", None)
    return _backend.doc_search(conn, query_vec, *args, **kwargs)


def doc_delete(conn, doc_id, **kwargs):
    """仅 PG 后端支持带 ACL 的文档删除；SQLite 脚手架按 id 删除。"""
    if not _use_pg:
        return _sqlite_doc_delete(conn, doc_id)
    return pg_store.doc_delete(conn, doc_id, **kwargs)


def _sqlite_doc_delete(conn, doc_id: int) -> int:
    cur = conn.execute("DELETE FROM docs WHERE id=?", (doc_id,))
    conn.commit()
    return cur.rowcount
