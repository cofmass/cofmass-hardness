"""CLI 运维入口（由原 app.py 降级）：python cli.py <cmd>。

正式对外入口是 FastAPI：uvicorn app:app。CLI 仅用于本机运维/调试。
"""
from __future__ import annotations

import argparse
import json

import storage as store
from config import EMB_MODEL
from graph.builder import build_agent
from graph.nodes import get_skills, set_skills
from providers.factory import get_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent 运维 CLI（SQLite 脚手架 / PG 均可）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="幂等建表")

    p_ing = sub.add_parser("ingest", help="灌入一条 RAG 知识")
    p_ing.add_argument("text")
    p_ing.add_argument("--source")
    p_ing.add_argument("--user", default="u-1")

    p_skill = sub.add_parser("skills", help="设置用户 skill 偏好")
    p_skill.add_argument("--user", default="u-1")
    p_skill.add_argument("--lang", default="zh")
    p_skill.add_argument("--verbosity", default="concise", choices=["concise", "detailed"])
    p_skill.add_argument("--auto-invoke", action="store_true")
    p_skill.add_argument("--skills", default="")

    p_ask = sub.add_parser("ask", help="问一个问题（跑通完整图）")
    p_ask.add_argument("question")
    p_ask.add_argument("--user", default="u-1")
    p_ask.add_argument("--thread", default="t-1")

    p_hist = sub.add_parser("history", help="查看会话 checkpoint 历史")
    p_hist.add_argument("--thread", default="t-1")

    args = parser.parse_args()

    conn = store.connect()
    store.setup(conn)

    if args.cmd == "init":
        print("OK: 表已就绪", flush=True)

    elif args.cmd == "ingest":
        vec = get_embeddings().embed_query(args.text)
        doc_id = store.doc_add(
            conn, args.text, vec, EMB_MODEL,
            metadata={"user_id": args.user}, source=args.source,
        )
        print(f"OK: 已灌入 doc#{doc_id} (dim={len(vec)})", flush=True)

    elif args.cmd == "skills":
        enabled = [s.strip() for s in args.skills.split(",") if s.strip()]
        set_skills(
            args.user, store, conn,
            preferences={"language": args.lang, "verbosity": args.verbosity,
                         "auto_invoke": args.auto_invoke},
            enabled=enabled,
        )
        print("OK: skill 偏好已写入", flush=True)

    elif args.cmd == "ask":
        agent = build_agent(conn=conn)
        result = agent.invoke(args.question, user_id=args.user, thread_id=args.thread)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    elif args.cmd == "history":
        from storage.checkpointer import make_saver
        saver, _ = make_saver()
        agent = build_agent(conn=conn, saver=saver)
        for step, snap in enumerate(agent.history(args.thread), 1):
            formatted = [f"{m['type']}: {str(m.get('content'))[:60]}" for m in snap["messages"]]
            print(f"step {step}: " + " | ".join(formatted), flush=True)

    conn.close()


if __name__ == "__main__":
    main()
