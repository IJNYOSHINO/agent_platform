from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import AsyncExecuteRequest, ExecuteRequest, TaskResponse, TaskStatus, ToolInfo, ToolListResponse
from api.task_store import get_task_store
from core.registry import get_registry
from core.security import get_current_user
from core.models import User
from services.agent_service import get_agent_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/execute", response_model=TaskResponse, summary="Execute task synchronously")
async def execute_sync(request: ExecuteRequest, current_user: User = Depends(get_current_user)) -> TaskResponse:
    service = get_agent_service()
    try:
        return await service.run(request, owner_id=current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/execute/stream", summary="Execute task as SSE stream")
async def execute_stream(request: ExecuteRequest, current_user: User = Depends(get_current_user)) -> StreamingResponse:
    task_id = request.task_id or str(uuid.uuid4())
    store = get_task_store()
    existing = await store.get(task_id, owner_id=current_user.id)
    if existing is None:
        created = await store.create(
            task_id,
            owner_id=current_user.id,
            instruction=request.instruction,
            media_content=[item.to_message_part() for item in request.media],
        )
        if created is None:
            raise HTTPException(status_code=409, detail=f"Task '{task_id}' already belongs to another user.")
    service = get_agent_service()
    run_input = service.build_background_input(request, task_id, owner_id=current_user.id)

    async def event_stream():
        async for event in service.stream_existing(run_input):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/execute/async", response_model=TaskResponse, summary="Execute task asynchronously")
async def execute_async(
    request: AsyncExecuteRequest,
    current_user: User = Depends(get_current_user),
) -> TaskResponse:
    task_id = request.task_id or str(uuid.uuid4())
    store = get_task_store()

    if not await store.get(task_id, owner_id=current_user.id):
        created = await store.create(
            task_id,
            owner_id=current_user.id,
            instruction=request.instruction,
            media_content=[item.to_message_part() for item in request.media],
        )
        if created is None:
            raise HTTPException(status_code=409, detail=f"Task '{task_id}' already belongs to another user.")
    await store.update(task_id, owner_id=current_user.id, status=TaskStatus.PENDING)
    response = await store.get(task_id, owner_id=current_user.id)
    if response is None:
        raise RuntimeError(f"Task '{task_id}' was not created.")
    return response


@router.get("/tasks/{task_id}", response_model=TaskResponse, summary="Get task status")
async def get_task(task_id: str, current_user: User = Depends(get_current_user)) -> TaskResponse:
    store = get_task_store()
    result = await store.get(task_id, owner_id=current_user.id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return await get_agent_service().resume_messages(result, owner_id=current_user.id)


@router.delete("/tasks/{task_id}", summary="Delete task")
async def delete_task(task_id: str, current_user: User = Depends(get_current_user)) -> dict[str, str]:
    store = get_task_store()
    success = await store.delete(task_id, owner_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return {"status": "success", "message": f"Task '{task_id}' deleted."}


@router.get("/tools/list", response_model=ToolListResponse, summary="List tools")
async def list_tools() -> ToolListResponse:
    tools = [
        ToolInfo(name=item["name"], description=item["description"], args_schema=item.get("args_schema", {}))
        for item in get_registry().list_tools()
    ]
    return ToolListResponse(tools=tools, total=len(tools))
