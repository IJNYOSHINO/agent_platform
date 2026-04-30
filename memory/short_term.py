"""
短期记忆（Short-Term Memory）
基于 LangGraph Checkpointer 实现：
  - 开发环境：MemorySaver（内存）
  - 生产环境：SqliteSaver（持久化）
此模块提供便捷的 checkpoint 访问接口，实际 checkpointer 在 graph.py 中绑定。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    短期记忆的访问封装。
    每次访问时动态创建数据库连接，保证与 AsyncSqliteSaver 的兼容性。
    """

    def __init__(self) -> None:
        from config.settings import get_settings
        self.cfg = get_settings()

    async def get_state(self, task_id: str) -> dict[str, Any] | None:
        config = {"configurable": {"thread_id": task_id}}
        try:
            if self.cfg.env == "prod":
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from core.graph import build_graph
                async with AsyncPostgresSaver.from_conn_string(self.cfg.checkpoint_database_url) as checkpointer:
                    await checkpointer.setup()
                    graph = build_graph(checkpointer=checkpointer)
                    snapshot = await graph.aget_state(config)
            else:
                from langgraph.checkpoint.memory import MemorySaver
                from core.graph import build_graph
                checkpointer = MemorySaver()
                graph = build_graph(checkpointer=checkpointer)
                snapshot = await graph.aget_state(config)

            if snapshot is None:
                return None
            return snapshot.values
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to retrieve state for task '%s': %s", task_id, exc)
            return None

    async def list_history(self, task_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """列出任务的历史快照（用于调试/审计）。"""
        config = {"configurable": {"thread_id": task_id}}
        history = []
        try:
            if self.cfg.env == "prod":
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                from core.graph import build_graph
                async with AsyncPostgresSaver.from_conn_string(self.cfg.checkpoint_database_url) as checkpointer:
                    await checkpointer.setup()
                    graph = build_graph(checkpointer=checkpointer)
                    async for state in graph.aget_state_history(config):
                        history.append(state.values)
                        if len(history) >= limit:
                            break
            else:
                from langgraph.checkpoint.memory import MemorySaver
                from core.graph import build_graph
                checkpointer = MemorySaver()
                graph = build_graph(checkpointer=checkpointer)
                async for state in graph.aget_state_history(config):
                    history.append(state.values)
                    if len(history) >= limit:
                        break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list history for task '%s': %s", task_id, exc)
        return history
