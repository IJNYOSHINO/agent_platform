from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# =========================
# 全局单例（进程级）
# =========================
_global_memory: Optional["ShortTermMemory"] = None
_global_lock = asyncio.Lock()


async def get_short_term_memory() -> "ShortTermMemory":
    """获取全局单例（并发安全）"""
    global _global_memory

    if _global_memory is not None:
        return _global_memory

    async with _global_lock:
        if _global_memory is None:
            _global_memory = ShortTermMemory()

    return _global_memory


async def close_short_term_memory():
    """关闭全局单例"""
    global _global_memory

    async with _global_lock:
        if _global_memory is not None:
            await _global_memory.close()
            _global_memory = None


# =========================
# 核心类
# =========================
class ShortTermMemory:
    """
    短期记忆封装：

    特性：
    - 进程级单例（通过 get_short_term_memory）
    - 懒加载 graph
    - 并发安全初始化
    - 支持 hard close（关闭后不可复用）
    """

    def __init__(self) -> None:
        from config.settings import get_settings

        self.cfg = get_settings()
        self._checkpointer = None
        self._graph = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_not_closed(self):
        """防止关闭后继续使用"""
        if self._closed:
            raise RuntimeError(
                "ShortTermMemory already closed. "
                "Use get_short_term_memory() to get a new instance."
            )

    async def _get_graph(self):
        """获取 graph（懒加载 + 并发安全）"""
        await self._ensure_not_closed()

        # 快路径
        if self._graph is not None:
            return self._graph

        # 慢路径（加锁）
        async with self._init_lock:
            await self._ensure_not_closed()

            if self._graph is not None:
                return self._graph

            logger.info("Initializing ShortTermMemory graph...")

            from core.graph import build_graph

            if self.cfg.env in ("prod", "production"):
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                self._checkpointer = await AsyncPostgresSaver.from_conn_string(
                    self.cfg.checkpoint_database_url
                )
                await self._checkpointer.setup()

                logger.info("Postgres checkpointer initialized")
            else:
                from langgraph.checkpoint.memory import MemorySaver

                self._checkpointer = MemorySaver()

                logger.info("MemorySaver initialized (dev mode)")

            self._graph = build_graph(checkpointer=self._checkpointer)

            logger.info("Graph initialized successfully")

        return self._graph

    # =========================
    # 对外接口
    # =========================
    async def get_state(self, task_id: str) -> Optional[dict[str, Any]]:
        """获取任务最新状态"""
        await self._ensure_not_closed()

        config = {"configurable": {"thread_id": task_id}}

        try:
            graph = await self._get_graph()
            snapshot = await graph.aget_state(config)

            if snapshot is None or not snapshot.values:
                return None

            return snapshot.values

        except Exception as exc:
            logger.warning(
                "get_state_failed",
                extra={
                    "component": "memory",
                    "task_id": task_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            return None

    async def list_history(
        self, task_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """获取历史状态"""
        await self._ensure_not_closed()

        config = {"configurable": {"thread_id": task_id}}
        history: list[dict[str, Any]] = []

        try:
            graph = await self._get_graph()

            async for state in graph.aget_state_history(config):
                if state.values:
                    history.append(state.values)

                if len(history) >= limit:
                    break

        except Exception as exc:
            logger.warning(
                "list_history_failed",
                extra={
                    "component": "memory",
                    "task_id": task_id,
                    "error": str(exc),
                },
            )

        return history

    async def close(self):
        """关闭资源（Hard close）"""
        if self._closed:
            return

        async with self._init_lock:
            if self._closed:
                return

            self._closed = True

            if self._checkpointer:
                try:
                    if hasattr(self._checkpointer, "aclose"):
                        await self._checkpointer.aclose()
                        logger.info("Checkpointer closed")
                except Exception as exc:
                    logger.warning("Error while closing checkpointer: %s", exc)
                finally:
                    self._checkpointer = None
                    self._graph = None