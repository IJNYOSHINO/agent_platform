from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


from config.settings import get_settings
from core.prompt_store import render_prompt

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self) -> None:
        cfg = get_settings()
        self._llm = ChatOpenAI(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            temperature=0,
        )

    def _build_tools_desc(self, tools_info: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for tool in tools_info:
            schema = tool.get("args_schema", {})
            props = schema.get("properties", {})
            params = ", ".join(f"{name}: {meta.get('type', 'any')}" for name, meta in props.items())
            lines.append(f"- {tool['name']}({params}): {tool['description']}")
        return "\n".join(lines) if lines else "(no tools available)"

    async def plan(
        self,
        instruction: str,
        tools_info: list[dict[str, Any]],
        memory_context: str = "",
        media_content: list[dict[str, Any]] | None = None,
        history: list[BaseMessage] | None = None,
    ) -> "ExecutionPlan":
        from api.schemas import ExecutionPlan, PlanStep
        system_prompt = render_prompt(
            "planner_system",
            tools_desc=self._build_tools_desc(tools_info),
            memory_context=memory_context or "(none)",
        )

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        if history:
            messages.extend(history)
        human_content: str | list[dict[str, Any]] = instruction
        if media_content:
            human_content = [{"type": "text", "text": instruction}, *media_content]
        messages.append(HumanMessage(content=human_content))

        response = await self._llm.ainvoke(messages)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM did not return valid JSON, falling back to direct_response: %s", raw)
            data = [
                {
                    "id": "step_reply",
                    "tool": "direct_response",
                    "input": {"text": raw},
                    "depends_on": [],
                    "description": "Fallback response for non-JSON output",
                }
            ]

        if isinstance(data, list):
            steps_data = data
        elif isinstance(data, dict):
            if "steps" in data:
                steps_data = data["steps"]
            elif "id" in data and "tool" in data:
                steps_data = [data]
            else:
                raise ValueError(f"Unable to recognize plan format: {data}")
        else:
            raise ValueError(f"Unable to recognize plan format: {type(data)}")
        for step in steps_data:
            step["depends_on"] = [
                dep.split(".")[0] if "." in dep else dep
                for dep in step.get("depends_on", [])
            ]
        plan = ExecutionPlan(steps=[PlanStep(**step) for step in steps_data])
        logger.info("Plan generated with %d steps.", len(plan.steps))
        return plan
