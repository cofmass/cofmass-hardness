"""阶段一存储门面：图/服务只依赖本模块，不感知 PostgreSQL 或 SQLite。

PG_DSN 配置时走 PostgreSQL + pgvector（阶段一正式载体）；
否则走 SQLite 脚手架（便于本机无 PG 时跑通逻辑，不计入阶段一 DoD）。
"""
from __future__ import annotations

from config import use_postgres
from storage import pg_store, sqlite_store

_backend = pg_store if use_postgres() else sqlite_store

connect = _backend.connect
setup = _backend.setup
ns = _backend.ns
mem_put = _backend.mem_put
mem_get = _backend.mem_get
mem_search = _backend.mem_search
mem_delete = _backend.mem_delete
doc_add = _backend.doc_add
doc_search = _backend.doc_search

# SQLite 脚手架专有（CLI 运维 / 旧测试用），PG 模式下不暴露
history = getattr(_backend, "history", None)


def backend_name() -> str:
    return "postgres+pgvector" if use_postgres() else "sqlite(scaffold)"


def extension_version(conn) -> str | None:
    if use_postgres():
        return pg_store.extension_version(conn)
    return "n/a"
