"""启动检测：拒绝超级用户 / BYPASSRLS 连接（阶段四 §3.2）。

用法（独立运维命令，显式执行）：
    python scripts/check_app_role.py                 # 使用 config.PG_DSN
    python scripts/check_app_role.py --dsn postgresql://...

无 PG_DSN / 无依赖时返回明确提示，不伪报通过。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage.session_context import check_role  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    from config import PG_DSN

    parser = argparse.ArgumentParser(description="校验 PG 应用连接角色符合 RLS 要求")
    parser.add_argument("--dsn", default=PG_DSN, help="PG DSN，默认读 HARDNESS_PG_DSN")
    args = parser.parse_args(argv)
    if not args.dsn:
        print("SKIP：未配置 HARDNESS_PG_DSN（SQLite 脚手架无需 RLS 角色校验）")
        return 0
    try:
        from storage import pg_store
        conn = pg_store.connect(args.dsn)
    except Exception as exc:
        print(f"ERROR：无法连接 PG：{exc}", file=sys.stderr)
        return 2
    try:
        info = check_role(conn)
    except RuntimeError as exc:
        print(f"FAIL：{exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(f"OK：role={info['role']} superuser=False bypassrls=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
