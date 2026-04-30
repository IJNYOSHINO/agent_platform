"""
执行引擎（Executor）
职责：
  1. 按拓扑排序依次执行 Plan 中的步骤
  2. 执行前解析 $step_id.field 变量引用（jsonpath-ng）
  3. 超时控制（asyncio.wait_for）
  4. 指数退避重试（tenacity）
  5. 记录每步详细结果
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from jsonpath_ng import parse as jp_parse
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from api.schemas import ExecutionPlan, PlanStep, StepResult
from config.settings import get_settings
from core.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ── 变量解析 ──────────────────────────────────────────────

def _resolve_value(value: Any, context: dict[str, Any]) -> Any:
    """
    递归替换 value 中的 $step_id.field[.nested] 引用，支持字符串内嵌。
    context: {step_id: output_dict}
    """
    if isinstance(value, str):
        # 处理整个字符串就是 $xxx 的情况（保持原有逻辑）
        if value.startswith("$") and len(value) > 1:
            ref = value[1:]
            parts = ref.split(".", 1)
            step_id = parts[0]
            field_path = parts[1] if len(parts) > 1 else None
            if step_id in context:
                step_output = context[step_id]
                if field_path is None:
                    return step_output
                jp_expr = jp_parse(f"$.{field_path}")
                matches = jp_expr.find(step_output if isinstance(step_output, dict) else {"value": step_output})
                if matches:
                    return matches[0].value
            return value

        # 处理字符串中间包含 $step1.xxx 的情况
        import re
        def replace_var(match):
            step_id = match.group(1)
            field_path = match.group(2)
            if step_id in context:
                step_output = context[step_id]
                jp_expr = jp_parse(f"$.{field_path}")
                matches = jp_expr.find(step_output if isinstance(step_output, dict) else {"value": step_output})
                if matches:
                    return str(matches[0].value)
            return match.group(0)

        return re.sub(r'\$([a-zA-Z_][a-zA-Z0-9_]*)\.(\w[\w.]*)', replace_var, value)

    if isinstance(value, dict):
        return {k: _resolve_value(v, context) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_value(item, context) for item in value]

    return value


def _resolve_inputs(
    raw_inputs: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    return {k: _resolve_value(v, context) for k, v in raw_inputs.items()}


# ── 拓扑排序 ──────────────────────────────────────────────

def _topological_sort(steps: list[PlanStep]) -> list[PlanStep]:
    """Kahn 算法拓扑排序，保证依赖先于被依赖步骤执行。"""
    id_to_step = {s.id: s for s in steps}
    in_degree: dict[str, int] = {s.id: 0 for s in steps}
    dependents: dict[str, list[str]] = {s.id: [] for s in steps}

    for step in steps:
        for dep in step.depends_on:
            in_degree[step.id] += 1
            dependents[dep].append(step.id)

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    sorted_ids: list[str] = []

    while queue:
        sid = queue.pop(0)
        sorted_ids.append(sid)
        for nxt in dependents[sid]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if len(sorted_ids) != len(steps):
        raise ValueError("Plan 中存在循环依赖，无法执行。")

    return [id_to_step[sid] for sid in sorted_ids]


# ── 执行引擎 ──────────────────────────────────────────────

class Executor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        cfg = get_settings()
        self._timeout = cfg.step_timeout
        self._max_retries = cfg.step_max_retries
        self._base_delay = cfg.retry_base_delay

    async def execute(
        self,
        plan: ExecutionPlan,
        progress_callback: Any | None = None,
    ) -> list[StepResult]:
        """
        按拓扑顺序执行所有步骤。
        progress_callback(step_result): 每步完成后回调（可选）
        """
        sorted_steps = _topological_sort(plan.steps)
        context: dict[str, Any] = {}   # step_id -> output
        results: list[StepResult] = []

        for step in sorted_steps:
            resolved_input = _resolve_inputs(step.input, context)
            sr = StepResult(
                step_id=step.id,
                tool_name=step.tool,
                input=resolved_input,
            )
            await self._run_step(step, resolved_input, sr)
            context[step.id] = sr.output
            results.append(sr)
            if progress_callback:
                await progress_callback(sr)

        return results

    async def _run_step(
        self,
        step: PlanStep,
        resolved_input: dict[str, Any],
        sr: StepResult,
    ) -> None:
        tool = self._registry.get(step.tool)
        if tool is None:
            sr.status = "failed"
            sr.error = f"Tool '{step.tool}' not found in registry."
            logger.error(sr.error)
            return

        sr.status = "running"
        start = time.perf_counter()

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(
                    multiplier=self._base_delay, min=self._base_delay, max=30
                ),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    sr.output = await asyncio.wait_for(
                        self._invoke_tool(tool, resolved_input),
                        timeout=self._timeout,
                    )

            sr.status = "success"
        except asyncio.TimeoutError:
            sr.status = "failed"
            sr.error = f"Step '{step.id}' timed out after {self._timeout}s."
            logger.error(sr.error)
        except Exception as exc:  # noqa: BLE001
            sr.status = "failed"
            sr.error = str(exc)
            logger.exception("Step '%s' failed: %s", step.id, exc)
        finally:
            sr.duration_seconds = time.perf_counter() - start

    @staticmethod
    async def _invoke_tool(tool: Any, inputs: dict[str, Any]) -> Any:
        """统一调用工具（同步/异步兼容）。"""
        if asyncio.iscoroutinefunction(tool._run):  # type: ignore[attr-defined]
            return await tool.arun(inputs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: tool.run(inputs))
