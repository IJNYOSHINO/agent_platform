from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 全局单例
# ──────────────────────────────────────────────────────────
_global_short_term_memory: ShortTermMemory | None = None
_global_lock = asyncio.Lock()


async def get_short_term_memory() -> ShortTermMemory:
    global _global_short_term_memory

    if _global_short_term_memory is not None:
        return _global_short_term_memory

    async with _global_lock:
        if _global_short_term_memory is None:
            _global_short_term_memory = ShortTermMemory()
            logger.info("Short-term memory initialized.")

    return _global_short_term_memory


async def close_short_term_memory() -> None:
    global _global_short_term_memory

    async with _global_lock:
        if _global_short_term_memory is not None:
            await _global_short_term_memory.close()
            _global_short_term_memory = None
            logger.info("Short-term memory closed.")


# ──────────────────────────────────────────────────────────
# 核心类
# ──────────────────────────────────────────────────────────
class ShortTermMemory:
    """
    短期记忆封装，基于 LangGraph Checkpointer。

    特性：
      - 进程级单例（通过 get_short_term_memory 获取）
      - graph 懒加载，首次调用时初始化
      - 并发安全（_init_lock 保护初始化路径）
      - 支持 warmup 提前初始化，避免首次请求延迟
      - dev 环境使用 MemorySaver，prod 使用 AsyncPostgresSaver
      - 正确持有 AsyncPostgresSaver 上下文管理器，关闭时安全退出
    """

    def __init__(self) -> None:
        from config.settings import get_settings

        self._cfg = get_settings()
        self._checkpointer: Any = None
        self._checkpointer_ctx: Any = None  # 持有 Postgres 异步上下文管理器
        self._graph: Any = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    # ── 内部工具 ──────────────────────────────────────────

    async def _ensure_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("ShortTermMemory is already closed.")

    async def _get_graph(self) -> Any:
        """获取 graph（懒加载 + 并发安全）。"""
        await self._ensure_not_closed()

        if self._graph is not None:
            return self._graph

        async with self._init_lock:
            await self._ensure_not_closed()

            if self._graph is not None:
                return self._graph

            logger.info("Initializing ShortTermMemory graph...")
            from core.graph import build_graph

            if self._cfg.env in ("prod", "production"):
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                # from_conn_string 返回异步上下文管理器，不能直接 await
                self._checkpointer_ctx = AsyncPostgresSaver.from_conn_string(
                    self._cfg.checkpoint_database_url
                )
                self._checkpointer = await self._checkpointer_ctx.__aenter__()
                await self._checkpointer.setup()
                logger.info("Postgres checkpointer initialized.")
            else:
                from langgraph.checkpoint.memory import MemorySaver

                self._checkpointer = MemorySaver()
                logger.info("MemorySaver initialized (dev mode).")

            self._graph = build_graph(checkpointer=self._checkpointer)
            logger.info("ShortTermMemory graph initialized.")

        return self._graph

    # ── 对外接口 ──────────────────────────────────────────

    async def warmup(self) -> bool:
        """提前初始化 graph，避免首次请求延迟。"""
        try:
            await self._get_graph()
            logger.info("ShortTermMemory warmup completed.")
            return True
        except Exception:
            logger.exception("ShortTermMemory warmup failed.")
            return False

    async def get_state(self, task_id: str) -> dict[str, Any] | None:
        """获取任务最新状态快照。"""
        await self._ensure_not_closed()

        config = {"configurable": {"thread_id": task_id}}
        try:
            graph = await self._get_graph()
            snapshot = await graph.aget_state(config)
            if snapshot is None or not snapshot.values:
                return None
            return snapshot.values
        except Exception:
            logger.warning("Failed to get state for task '%s'.", task_id, exc_info=True)
            return None

    async def list_history(
        self,
        task_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """获取任务的历史状态列表。"""
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
        except Exception:
            logger.warning("Failed to list history for task '%s'.", task_id, exc_info=True)

        return history

    # ── 关闭 ──────────────────────────────────────────────

    async def close(self) -> None:
        if self._closed:
            return

        async with self._init_lock:
            if self._closed:
                return

            self._closed = True

            if self._checkpointer is not None:
                try:
                    # MemorySaver 有 aclose，AsyncPostgresSaver 通过上下文管理器退出
                    if hasattr(self._checkpointer, "aclose"):
                        await self._checkpointer.aclose()

                    # 退出 Postgres 异步上下文管理器，释放连接池
                    if self._checkpointer_ctx is not None:
                        await self._checkpointer_ctx.__aexit__(None, None, None)

                    logger.info("Checkpointer closed.")
                except Exception:
                    logger.warning("Error while closing checkpointer.", exc_info=True)
                finally:
                    self._checkpointer = None
                    self._checkpointer_ctx = None
                    self._graph = None