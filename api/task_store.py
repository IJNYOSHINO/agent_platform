from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, TypedDict

from sqlalchemy import select

from api.schemas import TaskResponse, TaskStatus
from core.database import SessionLocal
from core.models import TaskRecord


class PendingTaskPayload(TypedDict):
    task_id: str
    owner_id: int | None
    instruction: str
    media_content: list[dict[str, Any]]


class TaskStore:
    """PostgreSQL-backed task status store."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @staticmethod
    def _record_query(task_id: str, owner_id: int | None = None):
        stmt = select(TaskRecord).where(TaskRecord.task_id == task_id)
        if owner_id is None:
            stmt = stmt.where(TaskRecord.owner_id.is_(None))
        else:
            stmt = stmt.where(TaskRecord.owner_id == owner_id)
        return stmt

    async def create(
        self,
        task_id: str,
        owner_id: int | None = None,
        instruction: str = "",
        media_content: list[dict[str, Any]] | None = None,
    ) -> TaskResponse | None:
        now = datetime.now(timezone.utc)
        async with self._lock:
            with SessionLocal() as db:
                existing = db.get(TaskRecord, task_id)
                if existing is None:
                    record = TaskRecord(
                        task_id=task_id,
                        owner_id=owner_id,
                        status=TaskStatus.PENDING.value,
                        instruction=instruction,
                        media_content=media_content or [],
                        plan=[],
                        steps_result=[],
                        result="",
                        error=None,
                        messages=[],
                        created_at=now,
                        updated_at=now,
                    )
                    db.add(record)
                    db.commit()
                elif owner_id is not None:
                    if existing.owner_id is None:
                        existing.owner_id = owner_id
                        if instruction:
                            existing.instruction = instruction
                        if media_content is not None:
                            existing.media_content = media_content
                        existing.updated_at = now
                        db.commit()
                    elif existing.owner_id != owner_id:
                        return None
                elif existing.owner_id is not None:
                    return None

        result = await self.get(task_id, owner_id=owner_id)
        if result is None:
            raise RuntimeError(f"Task '{task_id}' was not created.")
        return result

    async def update(self, task_id: str, owner_id: int | None = None, **kwargs: Any) -> None:
        if not kwargs:
            return

        async with self._lock:
            with SessionLocal() as db:
                record = db.scalar(self._record_query(task_id, owner_id))
                if record is None:
                    return

                for key, value in kwargs.items():
                    if key == "status" and isinstance(value, TaskStatus):
                        value = value.value
                    setattr(record, key, value)
                record.updated_at = datetime.now(timezone.utc)
                db.commit()

    async def claim_next_pending(self) -> PendingTaskPayload | None:
        async with self._lock:
            with SessionLocal() as db:
                record = db.scalar(
                    select(TaskRecord)
                    .where(TaskRecord.status == TaskStatus.PENDING.value)
                    .order_by(TaskRecord.created_at.asc())
                )
                if record is None:
                    return None
                record.status = TaskStatus.RUNNING.value
                record.updated_at = datetime.now(timezone.utc)
                db.commit()
                return {
                    "task_id": record.task_id,
                    "owner_id": record.owner_id,
                    "instruction": record.instruction or "",
                    "media_content": record.media_content or [],
                }

    async def get(self, task_id: str, owner_id: int | None = None) -> TaskResponse | None:
        with SessionLocal() as db:
            record = db.scalar(self._record_query(task_id, owner_id))
            if record is None:
                return None

            return TaskResponse(
                task_id=record.task_id,
                status=TaskStatus(record.status),
                plan=record.plan or [],
                steps_result=record.steps_result or [],
                result=record.result or "",
                error=record.error,
                messages=record.messages or [],
            )

    async def delete(self, task_id: str, owner_id: int | None = None) -> bool:
        async with self._lock:
            with SessionLocal() as db:
                record = db.scalar(self._record_query(task_id, owner_id))
                if record is None:
                    return False
                db.delete(record)
                db.commit()
                return True

    async def recover_running_tasks(self) -> int:
        async with self._lock:
            with SessionLocal() as db:
                records = db.scalars(select(TaskRecord).where(TaskRecord.status == TaskStatus.RUNNING.value)).all()
                count = 0
                for record in records:
                    record.status = TaskStatus.PENDING.value
                    record.updated_at = datetime.now(timezone.utc)
                    count += 1
                if count:
                    db.commit()
                return count


_store: TaskStore | None = None


def get_task_store() -> TaskStore:
    global _store
    if _store is None:
        _store = TaskStore()
    return _store
