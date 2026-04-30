from __future__ import annotations

import asyncio
import logging

from api.task_store import TaskStore, get_task_store
from services.agent_service import AgentRunInput, AgentService, get_agent_service

logger = logging.getLogger(__name__)


class TaskRunner:
    def __init__(self, service: AgentService, store: TaskStore, poll_interval: float = 1.0) -> None:
        self._service = service
        self._store = store
        self._poll_interval = poll_interval

    async def run_pending_once(self) -> int:
        processed = 0
        while True:
            payload = await self._store.claim_next_pending()
            if payload is None:
                break
            processed += 1
            run_input = AgentRunInput(**payload)
            await self._service.run_existing(run_input)
        return processed

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        await self._store.recover_running_tasks()
        while not stop_event.is_set():
            processed = await self.run_pending_once()
            if processed == 0:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self._poll_interval)
                except asyncio.TimeoutError:
                    continue


_runner: TaskRunner | None = None


def get_task_runner() -> TaskRunner:
    global _runner
    if _runner is None:
        _runner = TaskRunner(service=get_agent_service(), store=get_task_store())
    return _runner
