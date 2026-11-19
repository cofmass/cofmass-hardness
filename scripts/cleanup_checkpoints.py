"""阶段二治理：checkpoint TTL 清理脚本（默认 dry-run）。

用法（显式参数化，符合 U-05）：
  python scripts/cleanup_checkpoints.py --retain-days 7            # 仅预览
  python scripts/cleanup_checkpoints.py --retain-days 7 --apply     # 真正删除

时间来源：langgraph 0.6.11 原生 checkpoints 表无墙钟时间列，db/migration_002.sql
为 checkpoints 增加 DB 触发器维护的 hardness_updated_at；本脚本按该列过滤。
删除主表 checkpoint 时级联清理同 thread/checkpoint_ns/checkpoint_id 的
checkpoint_writes（避免孤儿行）。checkpoint_blobs 按 channel/version 独立管理，
无 checkpoint_id 关联列，不属于单条 checkpoint 的级联范围（由 langgraph 自行维护）。
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 允许 `python scripts/cleanup_checkpoints.py` 直接运行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import use_postgres

TIME_COLUMN = "hardness_updated_at"


def _pg_candidates(conn, cutoff: datetime) -> list[tuple[str, str, str]]:
    """返回 [(thread_id, checkpoint_ns, checkpoint_id), ...]（过期主表 checkpoint）。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='checkpoints' AND column_name=%s",
            (TIME_COLUMN,),
        )
        if not cur.fetchone():
            raise RuntimeError(
                f"checkpoints 表缺少 {TIME_COLUMN} 列；请先执行 db/migration_002.sql"
            )
        cur.execute(
            f"SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints "
            f"WHERE {TIME_COLUMN} < %s",
            (cutoff,),
        )
        return [(r["thread_id"], r["checkpoint_ns"], r["checkpoint_id"]) for r in cur.fetchall()]


def _sqlite_candidates(conn, cutoff: datetime) -> list[tuple[str, int]]:
    cutoff_s = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute(
        "SELECT thread_id, step FROM checkpoints WHERE updated_at < ?", (cutoff_s,)
    ).fetchall()
    return [(r["thread_id"], r["step"]) for r in rows]


def _delete_pg(conn, candidates: list[tuple[str, str, str]]) -> int:
    deleted = 0
    for thread_id, ns, cid in candidates:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id=%s AND checkpoint_ns=%s AND checkpoint_id=%s",
                (thread_id, ns, cid),
            )
            deleted += cur.rowcount
            cur.execute(
                "DELETE FROM checkpoints WHERE thread_id=%s AND checkpoint_ns=%s AND checkpoint_id=%s",
                (thread_id, ns, cid),
            )
            deleted += cur.rowcount
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="清理过期 checkpoint（默认 dry-run）")
    parser.add_argument("--retain-days", type=int, required=True, help="保留最近 N 天，显式必填")
    parser.add_argument("--apply", action="store_true", help="显式确认后才真正删除")
    args = parser.parse_args()
    if args.retain_days < 0:
        parser.error("--retain-days 不能为负")

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.retain_days)
    pg = use_postgres()
    if pg:
        from storage import pg_store
        conn = pg_store.connect()
        candidates = _pg_candidates(conn, cutoff)
    else:
        from storage import sqlite_store
        conn = sqlite_store.connect()
        candidates = _sqlite_candidates(conn, cutoff)

    print(f"backend={'pg' if pg else 'sqlite'} cutoff={cutoff.isoformat()} "
          f"candidates={len(candidates)}")

    if not args.apply:
        for item in candidates[:20]:
            print(f"  [dry-run] would delete {item}")
        print("DRY-RUN：未执行删除。加 --apply 才生效。")
        conn.close()
        return 0

    if pg:
        deleted = _delete_pg(conn, candidates)
    else:
        deleted = 0
        for thread_id, step in candidates:
            cur = conn.execute(
                "DELETE FROM checkpoints WHERE thread_id=? AND step=?", (thread_id, step)
            )
            deleted += cur.rowcount
        conn.commit()
    print(f"APPLIED：删除 {deleted} 条 checkpoint 记录")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
