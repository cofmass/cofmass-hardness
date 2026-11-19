"""阶段一检查点：长生命周期 PostgresSaver 工厂。

关键约定（解决评审 blocking #1）：
- saver 在 FastAPI lifespan 中构造一次，供全应用复用；
- 禁止在请求内 `with` 包裹 saver 或关闭连接；连接只在 shutdown 关闭；
- 契约测试用 InMemorySaver 验证图结构，不依赖 PG。
"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from config import PG_DSN


def make_saver(dsn: str | None = None):
    """返回 (saver, close_fn)；PG 不可用时回退内存 saver（仅验证图）。"""
    if not PG_DSN:
        return InMemorySaver(), lambda: None

    from psycopg.rows import dict_row
    from langgraph.checkpoint.postgres import PostgresSaver
    from storage import pg_store

    conn = pg_store.connect(dsn or PG_DSN)
    saver = PostgresSaver(conn)
    saver.setup()
    return saver, conn.close


def make_async_saver(dsn: str | None = None):
    """阶段三：PG 下创建 AsyncPostgresSaver；不可用抛异常由调用方回退 to_thread。"""
    if not PG_DSN:
        return None, None

    import psycopg
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    conn = psycopg.AsyncConnection.connect(dsn or PG_DSN, autocommit=True)
    saver = AsyncPostgresSaver(conn)
    return saver, conn.close
