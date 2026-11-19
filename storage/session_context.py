"""请求内 RLS 会话上下文（阶段四 §3.2）。

在事务内 SET LOCAL app.user_id / app.tenant_id，事务结束自动还原，避免会话串号。
与阶段二应用层过滤形成双保险；应用连接必须是非超级用户、无 BYPASSRLS 角色。

用法：
    from storage.session_context import session_context
    with session_context(conn, user_id="u-1", tenant_id="t-1"):
        ...  # 同一事务内的读写被 RLS 限制为该 ACL

注意：本项目主连接为 autocommit 长生命周期连接。进入上下文会临时关闭 autocommit
开启事务，退出时提交并恢复原 autocommit 状态；SQLite（回退脚手架）下为 no-op。
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

log = logging.getLogger("hardness.storage.session_context")


def check_role(conn) -> dict:
    """校验当前 PG 连接角色；超级用户或 BYPASSRLS 抛 RuntimeError（复用于启动检测）。"""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT rolname, rolsuper, rolbypassrls
            FROM pg_roles WHERE rolname = current_user
        """)
        row = cur.fetchone()
    finally:
        close = getattr(cur, "close", None)
        if close:
            close()
    if row is None:
        raise RuntimeError("无法查询当前数据库角色")
    if isinstance(row, dict):
        name, superuser, bypass = row["rolname"], row["rolsuper"], row["rolbypassrls"]
    else:
        name, superuser, bypass = row[0], row[1], row[2]
    info = {"role": name, "superuser": bool(superuser), "bypassrls": bool(bypass)}
    if info["superuser"] or info["bypassrls"]:
        raise RuntimeError(
            f"DSN 使用了违规角色 {name!r}（superuser={info['superuser']}, "
            f"bypassrls={info['bypassrls']}）；RLS 形同虚设，拒绝启动。"
        )
    return info


def check_app_role_if_pg(connect_fn) -> dict | None:
    """PG 连接则返回角色检查结果，非 PG（sqlite）返回 None；违规抛 RuntimeError。

    以连接自身属性判定后端（与 session_context 一致），而非全局 config：
    测试用 sqlite 连接 + 配置了 PG DSN 时，不能误把 sqlite 当 PG 跑角色校验。
    """
    conn = connect_fn()
    try:
        # psycopg 连接暴露 autocommit 属性；sqlite3 连接没有 —— 据此识别 PG。
        if getattr(conn, "autocommit", None) is None:
            return None
        return check_role(conn)
    finally:
        conn.close()


def _escape(value: str) -> str:
    """SET LOCAL 参数值转义：拒绝 NUL，单引号翻倍。"""
    if "\x00" in value:
        raise ValueError("ACL 值不能包含 NUL 字符")
    return value.replace("'", "''")


@contextmanager
def session_context(conn, *, user_id: str | None, tenant_id: str | None = None) -> Iterator[None]:
    """为当前事务设置 RLS 需要的 app.* GUC；不支持的连接自动降级为 no-op。"""
    if not user_id:
        raise ValueError("user_id 必填（RLS 兜底依赖 app.user_id）")

    is_pg = bool(getattr(conn, "autocommit", None) is not None)
    if not is_pg:
        # SQLite 脚手架无 RLS：保持阶段二应用层过滤，不做任何 SET。
        yield
        return

    prev_autocommit = conn.autocommit
    entered = False
    try:
        conn.autocommit = False
        entered = True
        with conn.cursor() as cur:
            cur.execute("SET LOCAL app.user_id = %s", (_escape(user_id),))
            cur.execute("SET LOCAL app.tenant_id = %s",
                        (_escape(tenant_id) if tenant_id else None,))
        yield
        conn.commit()
    except Exception:
        if entered:
            try:
                conn.rollback()
            except Exception:
                log.exception("RLS 会话回滚失败")
        raise
    finally:
        if entered:
            try:
                conn.autocommit = prev_autocommit
            except Exception:
                log.exception("恢复 autocommit 失败")
