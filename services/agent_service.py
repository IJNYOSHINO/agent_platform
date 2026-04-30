from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import message_to_dict
from pydantic import BaseModel, ConfigDict, Field

from api.schemas import ExecuteRequest, TaskResponse, TaskStatus
from api.task_store import TaskStore, get_task_store
from core.graph import AgentGraph

logger = logging.getLogger(__name__)


class AgentRunInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    owner_id: int | None = None
    instruction: str = Field(min_length=1)
    media_content: list[dict[str, Any]] = Field(default_factory=list)


class AgentService:
    def __init__(self, agent: AgentGraph, store: TaskStore) -> None:
        self._agent = agent
        self._store = store

    async def start(self, request: ExecuteRequest, owner_id: int | None = None) -> TaskResponse:
        run_input = self._build_run_input(request, owner_id=owner_id)
        await self._ensure_task(run_input.task_id, run_input.instruction, run_input.media_content, owner_id=run_input.owner_id)
        await self._store.update(run_input.task_id, owner_id=run_input.owner_id, status=TaskStatus.PENDING)
        response = await self._store.get(run_input.task_id, owner_id=run_input.owner_id)
        if response is None:
            raise RuntimeError(f"Task '{run_input.task_id}' was not created.")
        return response

    async def run(self, request: ExecuteRequest, owner_id: int | None = None) -> TaskResponse:
        run_input = self._build_run_input(request, owner_id=owner_id)
        await self._ensure_task(run_input.task_id, run_input.instruction, run_input.media_content, owner_id=run_input.owner_id)
        await self._store.update(run_input.task_id, owner_id=run_input.owner_id, status=TaskStatus.RUNNING)
        await self.run_existing(run_input)
        result = await self._store.get(run_input.task_id, owner_id=run_input.owner_id)
        if result is None:
            raise RuntimeError(f"Task '{run_input.task_id}' disappeared during execution.")
        return result

    async def run_existing(self, run_input: AgentRunInput) -> None:
        await self._ensure_task(run_input.task_id, run_input.instruction, run_input.media_content, owner_id=run_input.owner_id)
        await self._store.update(run_input.task_id, owner_id=run_input.owner_id, status=TaskStatus.RUNNING)

        try:
            final_state = await self._agent.run(
                task_id=run_input.task_id,
                owner_id=run_input.owner_id,
                instruction=run_input.instruction,
                media_content=run_input.media_content,
            )
            await self._persist_final_state(run_input.task_id, final_state, owner_id=run_input.owner_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Task '%s' failed: %s", run_input.task_id, exc)
            await self._store.update(
                run_input.task_id,
                owner_id=run_input.owner_id,
                status=TaskStatus.FAILED,
                error=str(exc),
            )

    async def stream_existing(self, run_input: AgentRunInput) -> AsyncIterator[dict[str, Any]]:
        await self._ensure_task(run_input.task_id, run_input.instruction, run_input.media_content, owner_id=run_input.owner_id)
        await self._store.update(run_input.task_id, owner_id=run_input.owner_id, status=TaskStatus.RUNNING)
        yield {"event": "started", "task_id": run_input.task_id, "status": TaskStatus.RUNNING}

        try:
            async for update in self._agent.stream(
                task_id=run_input.task_id,
                owner_id=run_input.owner_id,
                instruction=run_input.instruction,
                media_content=run_input.media_content,
            ):
                if "plan" in update:
                    node_state = update["plan"]
                    yield {
                        "event": "plan",
                        "task_id": run_input.task_id,
                        "plan": node_state.get("plan", []),
                        "error": node_state.get("error"),
                    }
                elif "execute" in update:
                    node_state = update["execute"]
                    yield {
                        "event": "steps",
                        "task_id": run_input.task_id,
                        "steps_result": node_state.get("steps_result", []),
                        "result": node_state.get("last_result", ""),
                        "error": node_state.get("error"),
                    }
                elif "final" in update:
                    final_state = update["final"] or {}
                    await self._persist_final_state(run_input.task_id, final_state, owner_id=run_input.owner_id)
                    task = await self._store.get(run_input.task_id, owner_id=run_input.owner_id)
                    yield {
                        "event": "done",
                        "task_id": run_input.task_id,
                        "task": task.model_dump(mode="json") if task else None,
                    }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Streaming task '%s' failed: %s", run_input.task_id, exc)
            await self._store.update(
                run_input.task_id,
                owner_id=run_input.owner_id,
                status=TaskStatus.FAILED,
                error=str(exc),
            )
            yield {
                "event": "error",
                "task_id": run_input.task_id,
                "status": TaskStatus.FAILED,
                "error": str(exc),
            }

    async def resume_messages(self, task: TaskResponse, owner_id: int | None = None) -> TaskResponse:
        if task.messages:
            return task

        state = await self._agent.resume(task.task_id, owner_id=owner_id)
        if state and "messages" in state:
            task.messages = [message_to_dict(message) for message in state["messages"]]
        return task

    def build_background_input(self, request: ExecuteRequest, task_id: str, owner_id: int | None = None) -> AgentRunInput:
        return AgentRunInput(
            task_id=task_id,
            owner_id=owner_id,
            instruction=request.instruction,
            media_content=[item.to_message_part() for item in request.media],
        )

    async def _persist_final_state(
        self,
        task_id: str,
        final_state: dict[str, Any],
        owner_id: int | None = None,
    ) -> None:
        error = final_state.get("error")
        messages = final_state.get("messages", [])

        await self._store.update(
            task_id,
            owner_id=owner_id,
            status=TaskStatus.FAILED if error else TaskStatus.SUCCESS,
            plan=final_state.get("plan", []),
            steps_result=final_state.get("steps_result", []),
            result=final_state.get("last_result", ""),
            error=error,
            messages=[message_to_dict(message) for message in messages],
        )

    async def _ensure_task(
        self,
        task_id: str,
        instruction: str = "",
        media_content: list[dict[str, Any]] | None = None,
        owner_id: int | None = None,
    ) -> None:
        if await self._store.get(task_id, owner_id=owner_id):
            return

        created = await self._store.create(
            task_id,
            owner_id=owner_id,
            instruction=instruction,
            media_content=media_content,
        )
        if created is None:
            raise PermissionError(f"Task '{task_id}' already belongs to another user.")

    @staticmethod
    def _build_run_input(request: ExecuteRequest, owner_id: int | None = None) -> AgentRunInput:
        return AgentRunInput(
            task_id=request.task_id or str(uuid.uuid4()),
            owner_id=owner_id,
            instruction=request.instruction,
            media_content=[item.to_message_part() for item in request.media],
        )


_agent: AgentGraph | None = None
_service: AgentService | None = None


def get_agent() -> AgentGraph:
    global _agent
    if _agent is None:
        _agent = AgentGraph()
    return _agent


def get_agent_service() -> AgentService:
    global _service
    if _service is None:
        _service = AgentService(agent=get_agent(), store=get_task_store())
    return _service
