"""向量模型版本重建脚本（阶段四 §3.4）。

切换 embedding_model 后，旧向量维度/分布与新模型不兼容。检索层已按 embedding_model
过滤，因此不能改数覆盖；正确做法是「写入新版本向量 + 灰度切换 + 清理旧版本」。

用法（显式参数化，符合 U-05，默认 dry-run）：
  python scripts/rebuild_vectors.py --old-model nomic-embed-text \
      --new-model text-embedding-v3 --limit 100            # 预览
  python scripts/rebuild_vectors.py --old-model nomic-embed-text \
      --new-model text-embedding-v3 --limit 100 --apply     # 真正重建
  python scripts/rebuild_vectors.py --old-model nomic-embed-text --purge-old --apply

注意：--apply 会写入新模型向量（需新的 embedding 配置生效）；--purge-old 删除旧版本文档。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import use_postgres  # noqa: E402


def _pg_candidates(conn, old_model: str, limit: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, content FROM docs WHERE embedding_model=%s ORDER BY id LIMIT %s",
            (old_model, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def _sqlite_candidates(conn, old_model: str, limit: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, content FROM docs WHERE embedding_model=? ORDER BY id LIMIT ?",
        (old_model, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按 embedding_model 版本重建 / 清理向量")
    parser.add_argument("--old-model", required=True, help="旧 embedding_model 名称")
    parser.add_argument("--new-model", help="新 embedding_model；提供则重建该版本副本")
    parser.add_argument("--limit", type=int, default=100, help="本批处理上限（显式控制）")
    parser.add_argument("--purge-old", action="store_true", help="删除旧版本文档")
    parser.add_argument("--apply", action="store_true", help="显式确认后落库；否则只预览")
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit 必须为正整数")
    if not args.new_model and not args.purge_old:
        parser.error("至少提供 --new-model 或 --purge-old")

    pg = use_postgres()
    if pg:
        from storage import pg_store
        conn = pg_store.connect()
        rows = _pg_candidates(conn, args.old_model, args.limit)
    else:
        from storage import sqlite_store
        conn = sqlite_store.connect()
        rows = _sqlite_candidates(conn, args.old_model, args.limit)

    print(f"backend={'pg' if pg else 'sqlite'} old_model={args.old_model} "
          f"candidates={len(rows)} new_model={args.new_model} purge_old={args.purge_old}")
    for r in rows[:10]:
        print(f"  [candidate] id={r['id']} content={r['content'][:60]!r}")

    if not args.apply:
        print("DRY-RUN：未改动数据。加 --apply 才生效。")
        conn.close()
        return 0

    # 重建：逐条用新模型重新 embed（走 provider 工厂，图/存储零改动）。
    rebuilt = 0
    if args.new_model:
        from providers.factory import get_embeddings
        emb = get_embeddings()
        for r in rows:
            vec = emb.embed_query(r["content"])
            if pg:
                from storage import pg_store
                pg_store.doc_add(conn, r["content"], vec, args.new_model)
            else:
                from storage import sqlite_store
                sqlite_store.doc_add(conn, r["content"], vec, args.new_model)
            rebuilt += 1

    # 清理旧版本：只在显式 --purge-old 时删除，避免误删正在服务的版本。
    purged = 0
    if args.purge_old:
        if pg:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM docs WHERE embedding_model=%s AND id = ANY(%s)",
                            (args.old_model, [r["id"] for r in rows]))
                purged = cur.rowcount
        else:
            for r in rows:
                cur = conn.execute("DELETE FROM docs WHERE id=? AND embedding_model=?",
                                   (r["id"], args.old_model))
                purged += cur.rowcount
            conn.commit()

    print(f"APPLIED：rebuilt={rebuilt} purged={purged}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
