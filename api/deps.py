"""阶段二 ACL 上下文：统一从请求头提取 user_id/tenant_id，向下推到 Store 与 docs 过滤。

阶段一兼容：请求头缺省时回退到请求体字段（u-1 / 空租户）；因此契约不变。
阶段四可从 token 提取，此处接口保持不变。
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from config import STORE_FILTER_ENABLED


@dataclass(frozen=True)
class ACLContext:
    user_id: str
    tenant_id: str | None = None

    def metadata_filter(self) -> dict[str, str]:
        """docs.metadata JSONB 包含过滤条件；关闭总开关时返回空（不过滤）。"""
        if not STORE_FILTER_ENABLED:
            return {}
        f = {"user_id": self.user_id}
        if self.tenant_id:
            f["tenant_id"] = self.tenant_id
        return f

    def namespace(self, *parts: str) -> tuple[str, ...]:
        """Store namespace 首位锁 user_id；租户作为第二位（与阶段四 RLS 对齐）。"""
        if not STORE_FILTER_ENABLED:
            return tuple(parts)
        head = (self.user_id,) if not self.tenant_id else (self.tenant_id, self.user_id)
        return head + tuple(parts)


def acl_from_request(request: Request, *, body_user_id: str | None = None,
                     body_tenant_id: str | None = None) -> ACLContext:
    """请求头 x-user-id / x-tenant-id 优先；缺省回退请求体 / 默认值。"""
    user_id = request.headers.get("x-user-id") or body_user_id or "u-1"
    tenant_id = request.headers.get("x-tenant-id") or body_tenant_id or None
    return ACLContext(user_id=user_id, tenant_id=tenant_id)
