from __future__ import annotations

import asyncio
import logging
import os
import textwrap
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from sqlalchemy import delete, select, update

from config.settings import get_settings
from core.database import async_session_factory
from core.models import LongTermMemoryRecord

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. Long-term memory will be disabled.")


# ──────────────────────────────────────────────────────────
# 全局单例
# ──────────────────────────────────────────────────────────
_global_long_term_memory: _FaissLongTermMemory | _NoOpLongTermMemory | None = None
_global_lock = asyncio.Lock()


async def get_long_term_memory() -> _FaissLongTermMemory | _NoOpLongTermMemory:
    global _global_long_term_memory

    if _global_long_term_memory is not None:
        return _global_long_term_memory

    async with _global_lock:
        if _global_long_term_memory is None:
            if _FAISS_AVAILABLE:
                memory = _FaissLongTermMemory()
                await memory.recover_pending()
                _global_long_term_memory = memory
            else:
                _global_long_term_memory = _NoOpLongTermMemory()
            logger.info("Long-term memory initialized.")

    return _global_long_term_memory


async def close_long_term_memory() -> None:
    global _global_long_term_memory

    async with _global_lock:
        if _global_long_term_memory is not None:
            await _global_long_term_memory.close()
            _global_long_term_memory = None
            logger.info("Long-term memory closed.")


# ──────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────
class DashScopeEmbeddings:
    def __init__(self) -> None:
        cfg = get_settings()
        self.api_key = cfg.api_key
        self.api_url = cfg.embedding_api_url
        self.model = cfg.embedding_model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def embed_query(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "input": {"contents": [{"text": text}]},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = await self._get_client()
        resp = await client.post(self.api_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("output", {}).get("embeddings", [])
        if not embeddings:
            raise ValueError(f"No embeddings returned: {data}")
        return embeddings[0]["embedding"]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ──────────────────────────────────────────────────────────
# 文本处理工具
# ──────────────────────────────────────────────────────────
def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = [c.strip() for c in text.replace("\r", "\n").split("\n") if c.strip()]
    if chunks:
        return chunks
    return [p.strip() for p in textwrap.wrap(text, width=180) if p.strip()]


def _compress_text(text: str, max_chars: int = 900) -> str:
    lines = _split_sentences(text)
    if not lines:
        return ""
    res, total = [], 0
    for line in lines:
        line = " ".join(line.split())
        if not line:
            continue
        if total + len(line) > max_chars and res:
            break
        res.append(line)
        total += len(line)
        if total >= max_chars:
            break
    return "\n".join(res)[:max_chars].strip()


def _uuid_to_int64(record_id: str) -> int:
    """将 UUID 字符串稳定映射为非负 int64，用作 FAISS IndexIDMap 的 ID。"""
    return uuid.UUID(record_id).int & 0x7FFF_FFFF_FFFF_FFFF


# ──────────────────────────────────────────────────────────
# NoOp 实现（faiss 不可用时的降级）
# ──────────────────────────────────────────────────────────
class _NoOpLongTermMemory:
    @property
    def is_enabled(self) -> bool:
        return False

    async def warmup(self) -> bool:
        """NoOp 实现，始终返回成功。"""
        return True

    async def store(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def retrieve(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def reset(self) -> None:
        pass

    async def close(self) -> None:
        pass


# ──────────────────────────────────────────────────────────
# FAISS 实现
# ──────────────────────────────────────────────────────────
class _FaissLongTermMemory:
    """
    长期记忆，基于 FAISS + SQLAlchemy。

    特性：
      - IndexIDMap 绑定 UUID → int64，索引重建后对应关系不丢失
      - retrieve 批量 IN 查询，无 N+1 问题
      - flush_task 指数退避 + 连续失败上限，避免反复报错刷屏
      - close() 先等 flush_task 真正结束再写盘，无锁竞争
      - 启动时 recover_pending，崩溃后自动修复未完成记录
      - warmup() 主动探测 embedding 服务可达性，提前暴露配置错误
    """

    _FLUSH_INTERVAL = 5      # 正常 flush 间隔（秒）
    _FLUSH_MAX_FAILURES = 5  # 连续失败上限，超过后暂停 60s 再重试

    @property
    def is_enabled(self) -> bool:
        return True

    def __init__(self) -> None:
        cfg = get_settings()
        self._faiss_path = Path(cfg.faiss_index_path)
        self._faiss_path.parent.mkdir(parents=True, exist_ok=True)

        self._rebuild = bool(
            cfg.memory_rebuild
            or os.getenv("MEMORY_REBUILD", "").lower() in {"1", "true"}
        )

        self._dim: int | None = None
        self._index: Any = None
        self._faiss_lock = asyncio.Lock()
        self._dirty = False
        self._closed = False

        self._embeddings = DashScopeEmbeddings()
        self._flush_task: asyncio.Task | None = None

        self._load_index()
        self._start_flush_task()

    # ── 内部工具 ──────────────────────────────────────────

    async def _ensure_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("LongTermMemory is already closed.")

    def _make_index(self, dim: int) -> Any:
        return faiss.IndexIDMap(faiss.IndexFlatL2(dim))

    def _load_index(self) -> None:
        if self._rebuild:
            self._remove_index_file()
            return
        if self._faiss_path.exists():
            try:
                self._index = faiss.read_index(str(self._faiss_path))
                self._dim = self._index.d
                logger.info("FAISS index loaded: %d vectors.", self._index.ntotal)
            except Exception:
                logger.exception("Failed to load FAISS index, will start fresh.")
                self._index = None

    def _remove_index_file(self) -> None:
        if self._faiss_path.exists():
            self._faiss_path.unlink()
            logger.info("FAISS index file removed (rebuild mode).")

    def _add_to_index(self, vec: np.ndarray, faiss_id: int) -> None:
        """向索引添加一条向量（必须在 _faiss_lock 持有期间调用）。"""
        if self._index is None:
            self._dim = vec.shape[1]
            self._index = self._make_index(self._dim)
        self._index.add_with_ids(vec, np.array([faiss_id], dtype=np.int64))
        self._dirty = True

    # ── flush 任务 ────────────────────────────────────────

    def _start_flush_task(self) -> None:
        async def _loop() -> None:
            failures = 0
            delay = float(self._FLUSH_INTERVAL)
            while not self._closed:
                try:
                    await asyncio.sleep(delay)
                    await self._flush_index()
                    failures = 0
                    delay = float(self._FLUSH_INTERVAL)
                except asyncio.CancelledError:
                    break
                except Exception:
                    failures += 1
                    delay = min(self._FLUSH_INTERVAL * (2 ** failures), 120.0)
                    logger.exception(
                        "FAISS flush failed (attempt %d), retry in %.0fs.",
                        failures, delay,
                    )
                    if failures >= self._FLUSH_MAX_FAILURES:
                        logger.error(
                            "FAISS flush failed %d consecutive times, pausing 60s.",
                            failures,
                        )
                        await asyncio.sleep(60)
                        failures = 0
                        delay = float(self._FLUSH_INTERVAL)

        self._flush_task = asyncio.create_task(_loop())

    async def _flush_index(self) -> None:
        """持久化内存索引（仅当有脏数据时写盘）。"""
        async with self._faiss_lock:
            if self._index is not None and self._dirty:
                faiss.write_index(self._index, str(self._faiss_path))
                self._dirty = False
                logger.debug("FAISS index flushed to disk.")

    # ── 对外接口 ──────────────────────────────────────────

    async def warmup(self) -> bool:
        """
        验证 embedding 服务可达，提前暴露配置错误。

        索引加载和 recover_pending 已在 __init__ / get_long_term_memory
        中完成，warmup 的唯一职责是探测外部 embedding 服务连通性。
        """
        try:
            await self._embeddings.embed_query("warmup")
            logger.info("LongTermMemory warmup completed.")
            return True
        except Exception:
            logger.exception("LongTermMemory warmup failed.")
            return False

    async def store(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        await self._ensure_not_closed()

        compressed = _compress_text(text)
        if not compressed:
            return

        meta = metadata or {}
        owner_id = meta.get("owner_id")
        record_id = str(uuid.uuid4())
        faiss_id = _uuid_to_int64(record_id)

        # 步骤 1：写 DB pending（崩溃后 recover_pending 可重建）
        async with async_session_factory() as db:
            async with db.begin():
                db.add(LongTermMemoryRecord(
                    id=record_id,
                    text=text,
                    summary=compressed,
                    owner_id=owner_id,
                    metadata_json=meta,
                    faiss_id=faiss_id,
                    status="pending",
                ))

        # 步骤 2：生成 embedding
        try:
            vector = await self._embeddings.embed_query(compressed)
        except Exception:
            logger.exception("Embedding failed for record %s, skipping store.", record_id)
            return

        # 步骤 3：写入 FAISS
        try:
            async with self._faiss_lock:
                self._add_to_index(np.array([vector], dtype=np.float32), faiss_id)
        except Exception:
            logger.exception("FAISS add failed for record %s.", record_id)
            return

        # 步骤 4：标记 DB ready
        async with async_session_factory() as db:
            async with db.begin():
                await db.execute(
                    update(LongTermMemoryRecord)
                    .where(LongTermMemoryRecord.id == record_id)
                    .values(status="ready")
                )

        logger.debug("Stored memory record %s (faiss_id=%d).", record_id, faiss_id)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        await self._ensure_not_closed()

        try:
            vector = await self._embeddings.embed_query(query)
        except Exception:
            logger.exception("Embedding failed during retrieve.")
            return []

        vec_np = np.array([vector], dtype=np.float32)

        async with self._faiss_lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            k = min(top_k, self._index.ntotal)
            distances, faiss_ids = self._index.search(vec_np, k)

        valid_pairs: list[tuple[int, float]] = [
            (int(fid), float(dist))
            for fid, dist in zip(faiss_ids[0], distances[0])
            if fid != -1
        ]
        if not valid_pairs:
            return []

        valid_faiss_ids = [fid for fid, _ in valid_pairs]
        dist_map: dict[int, float] = {fid: dist for fid, dist in valid_pairs}

        async with async_session_factory() as db:
            stmt = select(LongTermMemoryRecord).where(
                LongTermMemoryRecord.faiss_id.in_(valid_faiss_ids),
                LongTermMemoryRecord.status == "ready",
            )
            if owner_id is not None:
                stmt = stmt.where(LongTermMemoryRecord.owner_id == owner_id)
            rows = (await db.execute(stmt)).scalars().all()

        results = [
            {
                "id": row.id,
                "text": row.text,
                "summary": row.summary,
                "score": -dist_map[row.faiss_id],
                "distance": dist_map[row.faiss_id],
            }
            for row in rows
        ]
        results.sort(key=lambda x: x["distance"])
        return results

    async def recover_pending(self) -> None:
        """启动时修复 pending 状态的记录，重建其 FAISS 向量。"""
        async with async_session_factory() as db:
            rows = (
                await db.execute(
                    select(LongTermMemoryRecord).where(
                        LongTermMemoryRecord.status == "pending"
                    )
                )
            ).scalars().all()

        if not rows:
            return

        logger.info("Recovering %d pending memory records...", len(rows))
        for r in rows:
            try:
                vec = await self._embeddings.embed_query(r.summary)
                async with self._faiss_lock:
                    self._add_to_index(np.array([vec], dtype=np.float32), r.faiss_id)

                async with async_session_factory() as db:
                    async with db.begin():
                        await db.execute(
                            update(LongTermMemoryRecord)
                            .where(LongTermMemoryRecord.id == r.id)
                            .values(status="ready")
                        )
                logger.debug("Recovered record %s.", r.id)
            except Exception:
                logger.exception("Failed to recover record %s, skipping.", r.id)

    async def reset(self) -> None:
        """清空所有记忆（索引文件 + 数据库记录）。"""
        async with self._faiss_lock:
            self._remove_index_file()
            self._index = None
            self._dirty = False

        async with async_session_factory() as db:
            async with db.begin():
                await db.execute(delete(LongTermMemoryRecord))

        logger.info("Long-term memory reset.")

    # ── 关闭 ──────────────────────────────────────────────

    async def close(self) -> None:
        """
        关闭顺序：
          1. 标记 closed，阻止新写入
          2. cancel flush_task 并等待其真正退出（释放锁）
          3. 最终 flush（无锁竞争）
          4. 关闭 embedding client
        """
        self._closed = True

        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except (asyncio.CancelledError, Exception):
                pass

        await self._flush_index()
        await self._embeddings.close()
        logger.info("Long-term memory closed and flushed.")