"""阶段三：LLM 自动抽取长期记忆候选 + 白名单/置信度/出现次数过滤。

设计红线（PHASE_3 §3.2 / §3.4）：
- 默认关闭，由 config.AUTO_MEMORY_ENABLED 控制；
- 写入 namespace 首位必须是当前 user_id（由 ACL 强注入，模型不得编造）；
- 只写白名单 area/key；置信度与出现次数不达标一律丢弃并记日志；
- 显式 PUT 的优先级高于自动写入：已有 updated_by != "auto" 的值不被覆盖。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage

from config import (
    AUTO_MEMORY_ENABLED,
    MEMORY_CONFIDENCE_THRESHOLD,
    MEMORY_OCCURRENCE_THRESHOLD,
    MEMORY_WHITELIST,
)
from graph.state import get_text_input as _input_text

log = logging.getLogger("hardness.memory_extractor")

_EXTRACT_PROMPT = """从用户最后一句话中抽取可长期记忆的偏好事实。
只输出 JSON 数组，无候选输出 []。每个元素形如：
{"area":"skills","key":"preferences","field":"language","value":"zh","confidence":0.9}
area/key 必须来自白名单 {whitelist}。不要编造 user_id 或 tenant_id。
"""


def _call_llm(llm, messages: list) -> str:
    resp = llm.invoke(messages)
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        content = "".join(parts)
    return str(content)


def _parse_candidates(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1] if raw.count("```") >= 2 else raw.strip("`")
    # 截取第一个 JSON 数组 / 对象
    start, end = raw.find("["), raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _namespace(state: dict, area: str) -> tuple[str, ...]:
    """ACL 强注入 namespace，模型提供的 user/tenant 一律忽略。"""
    user_id = state.get("user_id", "anonymous")
    tenant_id = state.get("tenant_id")
    head = (tenant_id, user_id) if tenant_id else (user_id,)
    return head + (area,)


def _store_get(rt: dict, namespace: tuple[str, ...], key: str):
    lg = rt.get("langgraph_store")
    if lg is not None:
        item = lg.get(namespace, key)
        return item.value if item is not None else None
    import json as _json
    from graph.nodes import _store_get as nodes_get
    return nodes_get(rt, namespace, key)


def _store_put(rt: dict, namespace: tuple[str, ...], key: str, value: dict) -> None:
    from graph.nodes import _store_put as nodes_put
    nodes_put(rt, namespace, key, value)


def extract_candidates(state: dict, *, runtime: dict | None = None) -> list[dict]:
    """返回通过白名单/置信度/出现次数校验的候选（不写库）。"""
    rt = runtime or {}
    text = _input_text(state)
    if not AUTO_MEMORY_ENABLED or not text.strip() or text.startswith("/remember "):
        return []
    llm = rt.get("llm")
    if llm is None:
        return []
    # MEMORY_WHITELIST 的值是 set（用于 O(1) 查找），json 不支持 —— 序列化前转 list。
    whitelist_json = json.dumps(
        {area: sorted(keys) for area, keys in MEMORY_WHITELIST.items()},
        ensure_ascii=False,
    )
    # 模板里含 JSON 示例的大括号，不能用 str.format；用 replace 精确替换占位符。
    prompt = _EXTRACT_PROMPT.replace("{whitelist}", whitelist_json)
    try:
        raw = _call_llm(llm, [SystemMessage(content=prompt)] + list(state.get("messages", [])))
        candidates = _parse_candidates(raw)
    except Exception as exc:
        log.warning("auto memory extraction failed: %s", exc)
        return []

    accepted: list[dict] = []
    for c in candidates:
        area, key = str(c.get("area", "")), str(c.get("key", ""))
        if area not in MEMORY_WHITELIST or key not in MEMORY_WHITELIST.get(area, set()):
            log.info("auto memory dropped by whitelist: area=%s key=%s", area, key)
            continue
        try:
            confidence = float(c.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < MEMORY_CONFIDENCE_THRESHOLD:
            log.info("auto memory dropped by confidence: %s", confidence)
            continue
        # 出现次数门槛：同 field+value 需达到阈值才写；例外是语言类偏好（确定性事实）。
        field, value = str(c.get("field", "")), c.get("value")
        ns = _namespace(state, area)
        old = _store_get(rt, ns, key) or {}
        if old.get("updated_by") not in (None, "auto"):
            log.info("auto memory skipped: explicit value exists for %s/%s", area, key)
            continue
        occurrences = int(old.get("_occurrences", {}).get(f"{field}={value}", 0)) + 1 if isinstance(old.get("_occurrences"), dict) else 1
        if field not in ("language",) and occurrences < MEMORY_OCCURRENCE_THRESHOLD and not old:
            log.info("auto memory deferred: occurrence=%d field=%s", occurrences, field)
            # 记录occurrence供下次累积（不计入正式写入）
            _store_put(rt, ns, key, {
                **(old or {}),
                "_occurrences": {**(old.get("_occurrences", {}) if isinstance(old, dict) else {}),
                                 f"{field}={value}": occurrences},
                "updated_by": "auto",
                "confidence": confidence,
            })
            continue
        accepted.append({
            "area": area, "key": key, "field": field, "value": value,
            "confidence": confidence, "occurrences": occurrences,
        })
    return accepted


def persist_candidates(state: dict, candidates: list[dict], *, runtime: dict | None = None) -> dict:
    """把通过校验的候选合并写入 mem_store，返回 written 摘要。"""
    rt = runtime or {}
    written: dict[str, Any] = {}
    for c in candidates:
        ns = _namespace(state, c["area"])
        key = c["key"]
        old = _store_get(rt, ns, key) or {}
        if old.get("updated_by") not in (None, "auto"):
            continue  # 显式值优先
        merged = dict(old)
        merged[c["field"]] = c["value"]
        merged.update({
            "updated_by": "auto",
            "confidence": c["confidence"],
            "source_turn": state.get("thread_id", ""),
        })
        merged.pop("_occurrences", None)
        _store_put(rt, ns, key, merged)
        written[f"{c['area']}/{key}"] = {c["field"]: c["value"]}
    return written


def extract_memory_node(state: dict, *, runtime: dict | None = None) -> dict:
    """图节点：抽取候选 → 校验 → 持久化。"""
    candidates = extract_candidates(state, runtime=runtime)
    written = persist_candidates(state, candidates, runtime=runtime)
    return {"memory_candidates": candidates, "written": written}
