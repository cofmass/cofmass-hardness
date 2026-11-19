"""阶段三异步执行入口：SSE 流式事件源。

设计（PHASE_3 §2）：
- `graph.a_stream_events()` 用 `graph.astream(stream_mode=["updates","messages"])` 产出
  严格有序事件：token（增量正文）→ tool_start → tool_end → done；
- PG 环境优先用 AsyncPostgresSaver（真异步 checkpoint）；若不可用，
  回退 `asyncio.to_thread(sync_agent.invoke)` —— 此时仅产出 done 事件（无 token/tool 增量），
  在 done.payload.degraded 标记，路由层可据此告知客户端降级。
- 事件不依赖具体 SDK；`graph/` 仅经本模块与 providers/storage 抽象交互。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from config import ASYNC_ENABLED, PG_DSN
from graph.builder import Agent, build_agent
from graph.state import AgentState

log = logging.getLogger("hardness.async_builder")


def _make_invoke_config(thread_id: str, user_id: str, tenant_id: str | None,
                        runtime: dict | None = None) -> dict:
    configurable = {"thread_id": thread_id, "user_id": user_id, "tenant_id": tenant_id}
    if runtime is not None:
        configurable["runtime"] = runtime
    return {"configurable": configurable}


def _extract_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


def _event(event: str, payload: dict[str, Any]) -> dict:
    return {"event": event, "payload": payload}


class AsyncAgent:
    """包装同步 Agent：优先图级 astream，不可用时 to_thread 回退。"""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.mode = "graph_astream"
        self._async_close = None
        if ASYNC_ENABLED:
            self._try_async_saver()
        else:
            self.mode = "to_thread"

    def _try_async_saver(self) -> None:
        if not PG_DSN:
            # SQLite 回退：sync Agent 的 InMemorySaver 图可安全 astream（无 I/O 阻塞）。
            self.mode = "graph_astream"
            return
        try:
            from storage.checkpointer import make_async_saver

            saver, close_fn = make_async_saver()
            self.async_saver = saver
            self._async_close = close_fn
            self.mode = "graph_astream_async_saver"
        except Exception as exc:
            log.warning("AsyncPostgresSaver unavailable, fallback to to_thread: %s", exc)
            self.mode = "to_thread"

    @property
    def degraded(self) -> bool:
        return self.mode == "to_thread"

    async def stream_events(self, question: str, user_id: str = "u-1",
                            thread_id: str = "t-1",
                            tenant_id: str | None = None,
                            config: dict | None = None) -> AsyncIterator[dict]:
        """产出有序 SSE 事件：token* → tool_start/tool_end* → done。"""
        if self.mode == "to_thread":
            async for ev in self._fallback_events(question, user_id, thread_id, tenant_id):
                yield ev
            return

        from tools.registry import bind as bind_tools_with_acl
        graph = self.agent._graph_for(user_id, tenant_id)
        # generate 用绑定工具做 schema；ToolNode 的绑定工具来自编译图，两者名称一致。
        invoke_config = _make_invoke_config(thread_id, user_id, tenant_id, self.agent.runtime)
        if config:
            for k, v in config.items():
                if k == "configurable":
                    invoke_config["configurable"].update(v)
                else:
                    invoke_config[k] = v
        state = {
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
            "tenant_id": tenant_id,
            "thread_id": thread_id,
            "context": [], "skill_ctx": {}, "answer": "", "written": {},
        }
        answer_parts: list[str] = []
        tool_results: list[dict] = []
        try:
            async for chunk in graph.astream(
                state, config=invoke_config,
                stream_mode=["updates", "messages"],
            ):
                mode, data = chunk if isinstance(chunk, tuple) and len(chunk) == 2 else ("updates", chunk)
                if mode == "messages":
                    msg, metadata = data if isinstance(data, tuple) else (data, None)
                    if isinstance(msg, AIMessageChunk):
                        delta = _extract_content(msg.content)
                        if delta and not getattr(msg, "tool_call_chunks", None):
                            answer_parts.append(delta)
                            yield _event("token", {"text": delta, "thread_id": thread_id})
                    continue
                # updates：节点级状态；捕获工具结果与最终答案（兜底）。
                if isinstance(data, dict):
                    for node_name, update in data.items():
                        if not isinstance(update, dict):
                            continue
                        if node_name == "tools":
                            for r in update.get("tool_results", []):
                                tool_results.append(r)
                                yield _event("tool_end", {
                                    "name": r.get("name"), "status": r.get("status"),
                                    "result": r.get("result"),
                                })
                        elif node_name == "generate":
                            for c in update.get("tool_calls", []) or []:
                                yield _event("tool_start", {
                                    "name": c.get("name"), "args": c.get("args", {}),
                                })
                            final = update.get("answer", "")
                            if final and not answer_parts:
                                answer_parts.append(final)
                                yield _event("token", {"text": final, "thread_id": thread_id})
                        yield _event("node", {"name": node_name, "status": "end",
                                              "thread_id": thread_id})
        except Exception as exc:
            log.exception("astream failed, fallback to to_thread: %s", exc)
            async for ev in self._fallback_events(question, user_id, thread_id, tenant_id):
                yield ev
            return

        # 协议：answer 为正常结束事件；done 作为同义兼容事件保留，便于前端平滑迁移。
        payload = {
            "answer": "".join(answer_parts),
            "context": [],
            "skill_ctx": {},
            "thread_id": thread_id,
            "step": 0,
            "tool_results": tool_results,
            "degraded": self.degraded,
        }
        yield _event("answer", payload)
        yield _event("done", payload)

    async def _fallback_events(self, question: str, user_id: str, thread_id: str,
                               tenant_id: str | None) -> AsyncIterator[dict]:
        """异步 saver 不可用时：在线程池跑同步图，只产出 answer/done 事件。"""
        result = await asyncio.to_thread(
            self.agent.invoke, question, user_id=user_id,
            thread_id=thread_id, tenant_id=tenant_id,
        )
        payload = {
            "answer": result.get("answer", ""),
            "context": result.get("context", []),
            "skill_ctx": result.get("skill_ctx", {}),
            "thread_id": thread_id,
            "step": result.get("step", 0),
            "tool_results": result.get("tool_results", []),
            "degraded": True,
        }
        yield _event("answer", payload)
        yield _event("done", payload)

    async def aclose(self) -> None:
        if self._async_close is not None:
            try:
                result = self._async_close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass
            self._async_close = None
        self.async_saver = None


def make_async_agent(agent: Agent | None = None, **kwargs) -> AsyncAgent:
    return AsyncAgent(agent or build_agent(**kwargs))
