from __future__ import annotations

import logging
import os
import textwrap
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from sqlalchemy import select

from config.settings import get_settings
from core.database import SessionLocal
from core.models import LongTermMemoryRecord

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. Long-term memory will be disabled.")


class DashScopeEmbeddings:
    def __init__(self) -> None:
        cfg = get_settings()
        self.api_key = cfg.api_key
        self.api_url = cfg.embedding_api_url
        self.model = cfg.embedding_model

    async def embed_query(self, text: str) -> list[float]:
        payload = {
            "model": self.model,
            "input": {
                "contents": [{"text": text}]
            }
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("output", {}).get("embeddings", [])
            if not embeddings:
                raise ValueError(f"No embeddings returned from DashScope: {data}")
            return embeddings[0]["embedding"]


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = [chunk.strip() for chunk in text.replace("\r", "\n").split("\n") if chunk.strip()]
    if chunks:
        return chunks
    return [part.strip() for part in textwrap.wrap(text, width=180) if part.strip()]


def _compress_text(text: str, max_chars: int = 900) -> str:
    lines = _split_sentences(text)
    if not lines:
        return ""
    compressed: list[str] = []
    total = 0
    for line in lines:
        line = " ".join(line.split())
        if not line:
            continue
        if total + len(line) > max_chars and compressed:
            break
        compressed.append(line)
        total += len(line)
        if total >= max_chars:
            break
    return "\n".join(compressed)[:max_chars].strip()


class _NoOpLongTermMemory:
    @property
    def is_enabled(self) -> bool:
        return False

    async def store(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        return None

    async def retrieve(self, query: str, top_k: int = 5, owner_id: int | None = None) -> list[dict[str, Any]]:
        return []

    async def reset(self) -> None:
        return None


class _FaissLongTermMemory:
    @property
    def is_enabled(self) -> bool:
        return True

    def __init__(self) -> None:
        cfg = get_settings()
        self._faiss_path = Path(cfg.faiss_index_path)
        self._rebuild = bool(cfg.memory_rebuild or os.getenv("MEMORY_REBUILD", "").lower() in {"1", "true", "yes"})
        self._dim: int | None = None
        self._faiss_path.parent.mkdir(parents=True, exist_ok=True)
        self._embeddings = DashScopeEmbeddings()
        self._index: Any = self._load_or_create_index()
        if self._rebuild:
            with SessionLocal() as db:
                db.query(LongTermMemoryRecord).delete()
                db.commit()

    def _load_or_create_index(self) -> Any:
        if self._rebuild:
            self._remove_index_files()
            return None

        if self._faiss_path.exists():
            try:
                idx = faiss.read_index(str(self._faiss_path))
                self._dim = idx.d
                return idx
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load FAISS index, creating new: %s", exc)
        return None

    def _remove_index_files(self) -> None:
        try:
            if self._faiss_path.exists():
                self._faiss_path.unlink()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to remove FAISS index: %s", exc)

    def _save_index(self) -> None:
        if self._index:
            faiss.write_index(self._index, str(self._faiss_path))

    async def store(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        compressed = _compress_text(text)
        if not compressed:
            return

        payload = compressed
        meta = metadata or {}
        owner_id = meta.get("owner_id")
        if meta.get("result"):
            payload = _compress_text(
                f"用户: {meta.get('instruction', '')}\n回复: {meta.get('result', '')}"
            ) or compressed

        vector = await self._embeddings.embed_query(payload)
        vec_np = np.array([vector], dtype=np.float32)

        if self._index is None:
            self._dim = len(vector)
            self._index = faiss.IndexFlatL2(self._dim)
            logger.info("Initialized FAISS index with dimension: %d", self._dim)

        faiss_idx = self._index.ntotal
        self._index.add(vec_np)

        with SessionLocal() as db:
            db.add(
                LongTermMemoryRecord(
                    faiss_index=faiss_idx,
                    id=str(uuid.uuid4()),
                    owner_id=owner_id,
                    text=text,
                    summary=compressed,
                    metadata_json=meta,
                )
            )
            db.commit()

        self._save_index()

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        owner_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        vector = await self._embeddings.embed_query(query)
        vec_np = np.array([vector], dtype=np.float32)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(vec_np, k)

        results: list[dict[str, Any]] = []
        with SessionLocal() as db:
            for dist, idx in zip(distances[0], indices[0]):
                idx = int(idx)
                if idx < 0:
                    continue
                stmt = select(LongTermMemoryRecord).where(LongTermMemoryRecord.faiss_index == idx)
                if owner_id is None:
                    stmt = stmt.where(LongTermMemoryRecord.owner_id.is_(None))
                else:
                    stmt = stmt.where(LongTermMemoryRecord.owner_id == owner_id)
                row = db.scalar(stmt)
                if row:
                    results.append(
                        {
                            "id": row.id,
                            "text": row.text,
                            "summary": row.summary,
                            "faiss_index": row.faiss_index,
                            "owner_id": row.owner_id,
                            "metadata": row.metadata_json or {},
                            "score": float(dist),
                        }
                    )
        return results

    async def reset(self) -> None:
        self._remove_index_files()
        self._index = None
        self._dim = None
        with SessionLocal() as db:
            db.query(LongTermMemoryRecord).delete()
            db.commit()


@lru_cache(maxsize=1)
def get_long_term_memory() -> _FaissLongTermMemory | _NoOpLongTermMemory:
    if _FAISS_AVAILABLE:
        return _FaissLongTermMemory()
    return _NoOpLongTermMemory()


def LongTermMemory() -> _FaissLongTermMemory | _NoOpLongTermMemory:  # noqa: N802
    return get_long_term_memory()
