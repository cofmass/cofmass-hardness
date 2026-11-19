"""阶段二 PostgreSQL 存储：阶段一 SQL 封装 + PGStore(BaseStore) + 混合检索。

三载体物理隔离不变：checkpointer / mem_store / docs。
图与节点禁止直接 import psycopg / pgvector —— 只经 storage 门面或注入的 Store。
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Literal

from psycopg import Connection
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from config import (
    EMB_DIM,
    EMB_MODEL,
    HNSW_EF_SEARCH,
    HNSW_ITERATIVE_SCAN,
    HYBRID_RRF_K,
    HYBRID_SEARCH_ENABLED,
    HYBRID_TOKENIZER,
    PG_DSN,
    RAG_TOP_K,
)
from datetime import datetime, timezone

from langgraph.store.base import (
    BaseStore,
    Item,
    SearchItem,
    NOT_PROVIDED,
    NotProvided,
    PutOp,
    GetOp,
    SearchOp,
    ListNamespacesOp,
    Op,
    Result,
)

# langgraph 的哨兵 repr 为 NOT_GIVEN，实际导出名 NOT_PROVIDED；统一别名保持可读性。
NOT_GIVEN = NOT_PROVIDED

# ---------------- 连接 / schema（阶段一兼容）----------------


def connect(dsn: str | None = None) -> Connection:
    """建立长生命周期连接（autocommit + dict_row + vector 编解码注册）。"""
    conn = Connection.connect(dsn or PG_DSN, autocommit=True, row_factory=dict_row)
    try:
        register_vector(conn)
    except Exception:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
    return conn


def setup(conn: Connection) -> None:
    """幂等建表：阶段一 schema + 阶段二 migration_002。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "db"
    with conn.cursor() as cur:
        cur.execute((root / "schema.sql").read_text(encoding="utf-8"))
        migration = root / "migration_002.sql"
        if migration.exists():
            cur.execute(migration.read_text(encoding="utf-8"))


def extension_version(conn: Connection) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
        row = cur.fetchone()
    return row["extversion"] if row else None


# ---------------- 基础工具 ----------------


def ns(*parts: str) -> str:
    """namespace 落库为 JSON 数组字符串，杜绝 u-1 / u-10 前缀歧义。"""
    return json.dumps(list(parts), ensure_ascii=False)


def _ns_prefix(prefix: tuple[str, ...]) -> str:
    """JSON 数组前缀 LIKE 模式：精确匹配边界，避免 u-1 命中 u-10。"""
    head = json.dumps(list(prefix), ensure_ascii=False)
    # prefix=("u-1",) -> '["u-1",%' ；prefix=("u-1","skills") -> '["u-1", "skills"]%'
    return head[:-1] + ",%" if len(head) > 1 and head.endswith("]") else head + "%"


def _to_pgvector(vec: Iterable[float]):
    from pgvector import Vector

    values = [float(x) for x in vec]
    if len(values) != EMB_DIM:
        raise ValueError(f"向量维度不一致: {len(values)}，期望 {EMB_DIM}")
    return Vector(values)


def _apply_hnsw_settings(conn: Connection) -> None:
    """pgvector>=0.8 迭代扫描避免 HNSW 带 metadata 过滤时提前截断；失败静默降级。"""
    if not HNSW_ITERATIVE_SCAN or HNSW_ITERATIVE_SCAN == "none":
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SET hnsw.iterative_scan = %s", (HNSW_ITERATIVE_SCAN,))
            cur.execute("SET hnsw.ef_search = %s", (HNSW_EF_SEARCH,))
    except Exception:
        pass  # 低版本 / 参数未注册：回退到默认 ef_search，不阻塞检索


# ---------------- PGStore(BaseStore)：实测签名（langgraph 0.6.11）----------------


class PGStore(BaseStore):
    """langgraph 原生 Store；namespace=tuple[str,...]，JSON 数组字符串落库。

    阶段二只实现 BaseStore 四个抽象方法；ttl / refresh_ttl 用 mem_store.expires_at，
    filter 用 value @> jsonb；namespace 前缀用 JSON 数组边界 LIKE 防串号。
    """

    def __init__(self, conn: Connection, *, ttl_enabled: bool = False):
        self.conn = conn
        self.ttl_enabled = ttl_enabled

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_GIVEN,
    ) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        # 官方 NOT_GIVEN 表示「不改过期时间」；None 显式表示不过期；数值表示秒。
        if ttl is NOT_GIVEN or ttl is None:
            with self.conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO mem_store (namespace, key, value)
                       VALUES (%s, %s, %s::jsonb)
                       ON CONFLICT (namespace, key) DO UPDATE SET
                         value=EXCLUDED.value, updated_at=now()""",
                    (ns(*namespace), key, payload),
                )
            return
        from datetime import timedelta

        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO mem_store (namespace, key, value, expires_at)
                   VALUES (%s, %s, %s::jsonb, now() + %s::interval)
                   ON CONFLICT (namespace, key) DO UPDATE SET
                     value=EXCLUDED.value, updated_at=now(), expires_at=EXCLUDED.expires_at""",
                (ns(*namespace), key, payload, timedelta(seconds=float(ttl))),
            )

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT namespace, key, value, expires_at, updated_at FROM mem_store
                   WHERE namespace=%s AND key=%s
                     AND (expires_at IS NULL OR expires_at > now())""",
                (ns(*namespace), key),
            )
            row = cur.fetchone()
        if row is None:
            return None
        if refresh_ttl and row["expires_at"] is not None:
            # 续期：按原 ttl 长度延长（expires_at - now() + now() => 当前到期时间 + 原时长近似；
            # 简化为再延一个原剩余窗口，文档注明）。
            with self.conn.cursor() as cur:
                cur.execute(
                    """UPDATE mem_store SET expires_at = expires_at + (expires_at - now())
                       WHERE namespace=%s AND key=%s""",
                    (ns(*namespace), key),
                )
        ts = row["updated_at"] or datetime.now(timezone.utc)
        return Item(
            value=row["value"],
            key=row["key"],
            namespace=tuple(namespace),
            created_at=ts,
            updated_at=ts,
        )

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        pattern = _ns_prefix(namespace_prefix)
        sql = [
            "SELECT namespace, key, value, expires_at, updated_at FROM mem_store",
            "WHERE namespace LIKE %s AND (expires_at IS NULL OR expires_at > now())",
        ]
        params: list[Any] = [pattern]
        if filter:
            sql.append("AND value @> %s::jsonb")
            params.append(json.dumps(filter, ensure_ascii=False))
        sql.append("ORDER BY key LIMIT %s OFFSET %s")
        params.extend([limit, offset])
        with self.conn.cursor() as cur:
            cur.execute("\n".join(sql), params)
            rows = cur.fetchall()
        items: list[SearchItem] = []
        for r in rows:
            if refresh_ttl and r["expires_at"] is not None:
                with self.conn.cursor() as cur:
                    cur.execute(
                        """UPDATE mem_store SET expires_at = expires_at + (expires_at - now())
                           WHERE namespace=%s AND key=%s""",
                        (r["namespace"], r["key"]),
                    )
            items.append(
                SearchItem(
                    tuple(json.loads(r["namespace"])),
                    r["key"],
                    r["value"],
                    r["updated_at"] or datetime.now(timezone.utc),
                    r["updated_at"] or datetime.now(timezone.utc),
                )
            )
        return items

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM mem_store WHERE namespace=%s AND key=%s",
                (ns(*namespace), key),
            )

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """顺序执行 Op；BaseStore 抽象要求。阶段二节点只用单操作，批量按序兼容。"""
        results: list[Any] = []
        for op in ops:
            if isinstance(op, PutOp):
                self.put(op.namespace, op.key, op.value or {}, op.index, ttl=op.ttl)
                results.append(None)
            elif isinstance(op, GetOp):
                results.append(self.get(op.namespace, op.key, refresh_ttl=op.refresh_ttl))
            elif isinstance(op, SearchOp):
                results.append(
                    self.search(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                        refresh_ttl=op.refresh_ttl,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                with self.conn.cursor() as cur:
                    cur.execute("SELECT DISTINCT namespace FROM mem_store ORDER BY namespace")
                    results.append([tuple(json.loads(r["namespace"])) for r in cur.fetchall()])
            else:
                raise TypeError(f"不支持的 Store Op: {type(op)!r}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """异步批量：当前连接是同步的，直接委托 batch（阶段二无异步节点）。"""
        return self.batch(ops)


# ---------------- 阶段一 SQL 封装（回退 / 兼容性保留）----------------


def mem_put(conn: Connection, namespace: str, key: str, value: dict, *, ttl: float | None = None) -> None:
    PGStore(conn).put(tuple(json.loads(namespace)), key, value, ttl=ttl if ttl is not None else NOT_GIVEN)


def mem_get(conn: Connection, namespace: str, key: str) -> dict | None:
    item = PGStore(conn).get(tuple(json.loads(namespace)), key)
    return item.value if item else None


def mem_search(conn: Connection, namespace: str) -> dict[str, dict]:
    items = PGStore(conn).search(tuple(json.loads(namespace)))
    return {it.key: it.value for it in items}


def mem_delete(conn: Connection, namespace: str, key: str | None = None) -> int:
    with conn.cursor() as cur:
        if key is None:
            cur.execute("DELETE FROM mem_store WHERE namespace=%s", (namespace,))
        else:
            cur.execute("DELETE FROM mem_store WHERE namespace=%s AND key=%s", (namespace, key))
        return cur.rowcount


# ---------------- docs 检索：纯向量 + tsvector/RRF 混合 ----------------


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


def doc_delete(conn: Connection, doc_id: int, *, user_id: str | None = None, tenant_id: str | None = None) -> int:
    """带 ACL 校验的文档删除；调用方须先按当前上下文确认归属。"""
    acl: dict[str, str] = {}
    if user_id:
        acl["user_id"] = user_id
    if tenant_id:
        acl["tenant_id"] = tenant_id
    with conn.cursor() as cur:
        if acl:
            cur.execute(
                "DELETE FROM docs WHERE id=%s AND metadata @> %s::jsonb",
                (doc_id, json.dumps(acl, ensure_ascii=False)),
            )
        else:
            cur.execute("DELETE FROM docs WHERE id=%s", (doc_id,))
        return cur.rowcount


def doc_search(
    conn: Connection,
    query_vec: Iterable[float],
    top_k: int = RAG_TOP_K,
    metadata_filter: dict | None = None,
    embedding_model: str | None = EMB_MODEL,
    *,
    query_text: str | None = None,
    hybrid: bool | None = None,
) -> list[dict]:
    """向量检索；hybrid=True 时叠加 tsvector + RRF 融合（需 query_text）。"""
    use_hybrid = HYBRID_SEARCH_ENABLED if hybrid is None else hybrid
    if use_hybrid and query_text:
        return _hybrid_search(conn, query_vec, query_text, top_k, metadata_filter, embedding_model)
    qvec = _to_pgvector(query_vec)
    _apply_hnsw_settings(conn)
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
    return [_row_to_hit(r) for r in rows]


def _hybrid_search(
    conn: Connection,
    query_vec: Iterable[float],
    query_text: str,
    top_k: int,
    metadata_filter: dict | None,
    embedding_model: str | None,
) -> list[dict]:
    """pgvector + tsvector 双路召回，RRF 融合；候选集放大到 4*top_k 防截断。"""
    qvec = _to_pgvector(query_vec)
    _apply_hnsw_settings(conn)
    meta_json = json.dumps(metadata_filter or {}, ensure_ascii=False)
    k = HYBRID_RRF_K
    cand = max(top_k * 4, 20)
    sql = f"""
    WITH vec AS (
      SELECT id, row_number() OVER (ORDER BY embedding <=> %s::vector) AS r
      FROM docs
      WHERE metadata @> %s::jsonb AND embedding_model = %s
      ORDER BY embedding <=> %s::vector LIMIT %s
    ), ft AS (
      SELECT id, row_number() OVER (ORDER BY ts_rank_cd(content_tsv, plainto_tsquery(%s, %s)) DESC) AS r
      FROM docs
      WHERE metadata @> %s::jsonb AND embedding_model = %s
        AND content_tsv @@ plainto_tsquery(%s, %s)
      ORDER BY ts_rank_cd(content_tsv, plainto_tsquery(%s, %s)) DESC LIMIT %s
    )
    SELECT d.id, d.content, d.metadata, d.source,
           coalesce(1.0/(%s+vec.r),0) + coalesce(1.0/(%s+ft.r),0) AS score
    FROM docs d
    LEFT JOIN vec ON vec.id = d.id
    LEFT JOIN ft  ON ft.id = d.id
    WHERE vec.id IS NOT NULL OR ft.id IS NOT NULL
    ORDER BY score DESC LIMIT %s
    """
    params: list[Any] = [
        qvec, meta_json, embedding_model, qvec, cand,
        HYBRID_TOKENIZER, query_text,
        meta_json, embedding_model,
        HYBRID_TOKENIZER, query_text,
        HYBRID_TOKENIZER, query_text, cand,
        k, k, top_k,
    ]
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [_row_to_hit(r) for r in rows]


def _row_to_hit(r: dict) -> dict:
    return {
        "id": r["id"],
        "content": r["content"],
        "metadata": r["metadata"],
        "score": float(r["score"]),
        "source": r["source"],
    }
