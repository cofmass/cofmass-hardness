"""阶段四 §3.1 Token 预算与裁剪。

目标：上下文中毒 / 无限增长时守住预算，而不是靠「截断字符串」假装解决。
策略（按优先级从低到高丢弃）：
  1. 旧工具结果（tool_results 尾部优先保留）
  2. 旧检索片段（context 尾部优先保留）
  3. 最旧的人/机消息（保留 system 与最后一条 human）
  4. 仍超限：对最后一条 human 按字符硬截断并标记 truncated=True

计数：优先使用 tiktoken（若已安装）；否则启发式估算（中文按 1 字≈1.5 token，英文按 4 字符≈1 token），
并在响应中给出 estimated=True，便于运维识别精度。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import MAX_CONTEXT_TOKENS

log = logging.getLogger("hardness.token_budget")


@dataclass
class TrimResult:
    messages: list
    context: list[dict]
    tool_results: list[dict]
    truncated: bool = False
    estimated: bool = False
    original_tokens: int = 0
    final_tokens: int = 0

    def as_meta(self) -> dict:
        return {
            "truncated": self.truncated,
            "estimated": self.estimated,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
        }


def _tiktoken_encode(text: str):
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text)
    except Exception:
        return None


def count_tokens(text: str) -> tuple[int, bool]:
    """返回 (token 数, 是否估算)。"""
    if not text:
        return 0, False
    if _TIKTOKEN_AVAILABLE:
        tokens = _tiktoken_encode(text)
        if tokens is not None:
            return len(tokens), False
    # 启发式：CJK 单字按 2 token，其余按 4 字符≈1 token。
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, cjk * 2 + (other + 3) // 4), True


_TIKTOKEN_AVAILABLE = True
try:
    import tiktoken as _tiktoken_probe  # noqa: F401
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def _msg_text(msg) -> str:
    if isinstance(msg, dict):
        return str(msg.get("content", "") or "")
    return str(getattr(msg, "content", "") or "")


def _tools_text(tool_results: list[dict]) -> str:
    return "\n".join(str(t.get("result", "")) for t in tool_results)


def _context_text(context: list[dict]) -> str:
    return "\n".join(str(c.get("content", "")) for c in context)


def _sum(messages, context, tool_results) -> tuple[int, bool]:
    total = 0
    estimated = False
    texts = [_msg_text(m) for m in messages]
    texts += [_tools_text(tool_results), _context_text(context)]
    for text in texts:
        n, est = count_tokens(text)
        total += n
        estimated = estimated or est
    return total, estimated


def trim_to_budget(
    messages: list,
    *,
    context: list[dict] | None = None,
    tool_results: list[dict] | None = None,
    max_tokens: int | None = None,
) -> TrimResult:
    """按预算裁剪并返回结构化结果；绝不修改传入对象。"""
    budget = max_tokens if max_tokens is not None else MAX_CONTEXT_TOKENS
    if budget <= 0:
        raise ValueError("max_tokens 必须为正整数")

    msgs = list(messages)
    ctx = list(context or [])
    tools = list(tool_results or [])
    original, estimated = _sum(msgs, ctx, tools)
    if original <= budget:
        return TrimResult(msgs, ctx, tools, False, estimated, original, original)

    # 1) 旧工具结果：保留最近的最少条数，直到进预算。
    while tools and _sum(msgs, ctx, tools)[0] > budget:
        tools.pop(0)
    # 2) 旧检索片段：保留最近的。
    while ctx and _sum(msgs, ctx, tools)[0] > budget:
        ctx.pop(0)
    # 3) 旧消息：保留 system 与最后一条 human，其余从头丢。
    while len(msgs) > 2 and _sum(msgs, ctx, tools)[0] > budget:
        # 找第一条非 system、非最后一条 human 的消息。
        drop = None
        for i, m in enumerate(msgs[:-1]):
            typ = m.get("type") if isinstance(m, dict) else getattr(m, "type", "")
            if typ != "system":
                drop = i
                break
        if drop is None:
            break
        msgs.pop(drop)
    # 4) 最后一条 human 仍超预算：按字符比例硬截断（有损，标记 truncated）。
    total, _ = _sum(msgs, ctx, tools)
    truncated = False
    if total > budget and msgs:
        last = msgs[-1]
        text = _msg_text(last)
        if text:
            ratio = budget / max(total, 1)
            keep = max(16, int(len(text) * ratio))
            shortened = text[:keep] + "\n[已按 token 预算截断]"
            if isinstance(last, dict):
                last = {**last, "content": shortened}
            else:
                try:
                    last = last.model_copy(update={"content": shortened})
                except AttributeError:
                    last = type("TrimmedMessage", (), {
                        "type": getattr(last, "type", "human"),
                        "content": shortened,
                    })()
            msgs[-1] = last
            truncated = True

    final, est = _sum(msgs, ctx, tools)
    return TrimResult(msgs, ctx, tools, truncated, est or estimated, original, final)
