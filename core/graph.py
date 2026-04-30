from __future__ import annotations

import logging
import operator
from typing import Any, Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from api.schemas import ExecutionPlan, PlanStep
from config.settings import get_settings
from core.executor import Executor
from core.planner import Planner
from core.registry import get_registry
from memory.long_term import get_long_term_memory

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    instruction: str
    media_content: list[dict[str, Any]]
    owner_id: int | None
    plan: list[dict[str, Any]]
    steps_result: list[dict[str, Any]]
    last_result: str
    current_step: int
    error: str | None


CURRENT_DIALOG_MARKERS = (
    "刚刚",
    "刚才",
    "上一个",
    "上一轮",
    "刚问",
    "我问了",
    "还有吗",
    "我是谁",
    "你记得",
    "这个对话",
)

LONG_TERM_MARKERS = (
    "以前",
    "之前",
    "长期",
    "记忆里",
    "所有会话",
    "别的会话",
    "其他会话",
)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _is_current_dialog_query(instruction: str) -> bool:
    return any(marker in instruction for marker in CURRENT_DIALOG_MARKERS)


def _should_use_long_term_memory(instruction: str) -> bool:
    if _is_current_dialog_query(instruction):
        return False
    return any(marker in instruction for marker in LONG_TERM_MARKERS) or len(instruction) > 20


def _recent_messages(messages: list[BaseMessage], max_messages: int = 12) -> list[BaseMessage]:
    return messages[-max_messages:] if messages else []


def _format_short_term_history(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        text = _message_text(message).strip()
        if not text:
            continue
        if isinstance(message, HumanMessage):
            lines.append(f"用户: {text}")
        elif isinstance(message, AIMessage):
            lines.append(f"助手: {text}")
    return "\n".join(lines) or "(none)"


def _is_stable_memory_candidate(instruction: str, result: str) -> bool:
    text = f"{instruction}\n{result}"
    stable_markers = (
        "我叫",
        "我是",
        "我的名字",
        "你可以叫我",
        "记住",
        "偏好",
        "喜欢",
        "不喜欢",
    )
    volatile_markers = (
        "刚刚",
        "刚才",
        "上一个",
        "上一轮",
        "我问了",
        "还有吗",
        "这次",
        "当前",
    )
    return any(marker in text for marker in stable_markers) and not any(marker in text for marker in volatile_markers)


async def _store_long_term_summary(state: AgentState, owner_id: int | None = None) -> None:
    instruction = state.get("instruction", "").strip()
    last_result = state.get("last_result", "").strip()
    if not _is_stable_memory_candidate(instruction, last_result):
        return

    transcript = "\n".join(part for part in [f"用户: {instruction}", f"助手: {last_result}"] if part.strip())
    try:
        await get_long_term_memory().store(
            transcript,
            metadata={
                "source": "stable_user_memory",
                "instruction": instruction,
                "result": last_result,
                "owner_id": owner_id,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to store long-term memory: %s", exc)


async def _plan_node(state: AgentState) -> AgentState:
    cfg = get_settings()
    registry = get_registry()
    planner = Planner()
    instruction = state.get("instruction", "")
    recent = _recent_messages(state.get("messages", []))

    memory_context = ""
    if _should_use_long_term_memory(instruction):
        snippets = await get_long_term_memory().retrieve(
            instruction,
            top_k=cfg.memory_top_k,
            owner_id=state.get("owner_id"),
        )
        memory_context = _format_memory(snippets, cfg.memory_token_limit)

    short_term_prompt = (
        "Current conversation memory. Use this first for questions about the current session, "
        "such as who the user is, what they just asked, or what happened in the previous turn.\n"
        f"{_format_short_term_history(recent)}"
    )

    try:
        plan = await planner.plan(
            instruction=instruction,
            tools_info=registry.list_tools(),
            memory_context=memory_context,
            media_content=state.get("media_content", []) or None,
            history=[SystemMessage(content=short_term_prompt), *recent],
        )
        return {"plan": [step.model_dump() for step in plan.steps], "current_step": 0, "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Planner failed: %s", exc)
        return {"plan": [], "error": str(exc)}


async def _execute_node(state: AgentState) -> AgentState:
    plan_data = state.get("plan", [])
    if not plan_data:
        return {"steps_result": [], "last_result": "", "error": state.get("error")}

    plan = ExecutionPlan(steps=[PlanStep(**item) for item in plan_data])
    executor = Executor(get_registry())
    results = await executor.execute(plan)
    steps_result = [item.model_dump() for item in results]

    last_result = ""
    if results and results[-1].output is not None:
        last_result = str(results[-1].output)

    failed = [item for item in results if item.status == "failed"]
    error = "; ".join(item.error or "" for item in failed) if failed else None

    new_messages: list[BaseMessage] = []
    if last_result:
        new_messages.append(AIMessage(content=last_result))

    await _store_long_term_summary({**state, "last_result": last_result}, owner_id=state.get("owner_id"))

    return {
        "messages": new_messages,
        "steps_result": steps_result,
        "last_result": last_result,
        "error": error,
    }


def _should_execute(state: AgentState) -> str:
    if state.get("error") or not state.get("plan"):
        return END
    return "execute"


def _format_memory(snippets: list[dict[str, Any]], token_limit: int) -> str:
    if not snippets:
        return ""
    parts: list[str] = []
    total = 0
    for item in snippets:
        summary = item.get("summary") or item.get("text", "")
        estimated = max(1, len(summary) // 4)
        if total + estimated > token_limit:
            break
        parts.append(f"- {summary}")
        total += estimated
    return "\n".join(parts)


def build_graph(checkpointer: Any) -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("plan", _plan_node)
    workflow.add_node("execute", _execute_node)
    workflow.set_entry_point("plan")
    workflow.add_conditional_edges("plan", _should_execute, {"execute": "execute", END: END})
    workflow.add_edge("execute", END)
    return workflow.compile(checkpointer=checkpointer)


class AgentGraph:
    def __init__(self) -> None:
        self.cfg = get_settings()
        self._dev_checkpointer = MemorySaver()

    def _build_initial_state(self, instruction: str, media_content: list[dict[str, Any]] | None) -> dict[str, Any]:
        human_content: list[dict[str, Any]] | str = instruction
        if media_content:
            human_content = [{"type": "text", "text": instruction}, *media_content]
        return {
            "messages": [HumanMessage(content=human_content)],
            "instruction": instruction,
            "media_content": media_content or [],
            "owner_id": None,
            "plan": [],
            "steps_result": [],
            "last_result": "",
            "current_step": 0,
            "error": None,
        }

    async def run(
        self,
        task_id: str,
        owner_id: int | None,
        instruction: str,
        media_content: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        config = {"configurable": {"thread_id": task_id}}
        initial_state = self._build_initial_state(instruction, media_content)
        initial_state["owner_id"] = owner_id

        if self.cfg.env == "prod":
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with AsyncPostgresSaver.from_conn_string(self.cfg.checkpoint_database_url) as checkpointer:
                await checkpointer.setup()
                graph = build_graph(checkpointer=checkpointer)
                final_state = await graph.ainvoke(initial_state, config=config)
                return final_state  # type: ignore[return-value]

        graph = build_graph(checkpointer=self._dev_checkpointer)
        final_state = await graph.ainvoke(initial_state, config=config)
        return final_state  # type: ignore[return-value]

    async def stream(
        self,
        task_id: str,
        owner_id: int | None,
        instruction: str,
        media_content: list[dict[str, Any]] | None = None,
    ):
        config = {"configurable": {"thread_id": task_id}}
        initial_state = self._build_initial_state(instruction, media_content)
        initial_state["owner_id"] = owner_id

        if self.cfg.env == "prod":
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with AsyncPostgresSaver.from_conn_string(self.cfg.checkpoint_database_url) as checkpointer:
                await checkpointer.setup()
                graph = build_graph(checkpointer=checkpointer)
                async for update in graph.astream(initial_state, config=config, stream_mode="updates"):
                    yield update
                snapshot = await graph.aget_state(config)
                yield {"final": snapshot.values if snapshot else {}}
            return

        graph = build_graph(checkpointer=self._dev_checkpointer)
        async for update in graph.astream(initial_state, config=config, stream_mode="updates"):
            yield update
        snapshot = await graph.aget_state(config)
        yield {"final": snapshot.values if snapshot else {}}

    async def resume(self, task_id: str, owner_id: int | None = None) -> AgentState | None:
        config = {"configurable": {"thread_id": task_id}}
        if self.cfg.env == "prod":
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            async with AsyncPostgresSaver.from_conn_string(self.cfg.checkpoint_database_url) as checkpointer:
                await checkpointer.setup()
                graph = build_graph(checkpointer=checkpointer)
                snapshot = await graph.aget_state(config)
                if snapshot is None:
                    return None
                return snapshot.values  # type: ignore[return-value]

        graph = build_graph(checkpointer=self._dev_checkpointer)
        snapshot = await graph.aget_state(config)
        if snapshot is None:
            return None
        return snapshot.values  # type: ignore[return-value]
