"""
工具注册表（Tool Registry）
职责：
  1. 启动时扫描 /tools 目录下所有 .py 文件
  2. 自动发现并注册所有被 @tool 装饰的函数
  3. 提供全局工具查询接口
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """全局工具注册表（单例）"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ── 注册 ──────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("Tool '%s' already registered, overwriting.", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    # ── 扫描目录 ──────────────────────────────────────────

    def scan_directory(self, directory: str | Path) -> int:
        """
        扫描目录，动态加载所有 .py 文件中的 @tool 函数。
        返回本次注册的工具数量。
        """
        tools_dir = Path(directory)
        if not tools_dir.exists():
            logger.warning("Tools directory '%s' does not exist.", tools_dir)
            return 0

        count = 0
        for py_file in sorted(tools_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            count += self._load_module(py_file)

        logger.info("Tool scan complete. Total tools registered: %d", len(self._tools))
        return count

    def _load_module(self, path: Path) -> int:
        """加载单个模块，返回从该模块注册的工具数。"""
        module_name = f"_agent_tools.{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return 0
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load tool module '%s': %s", path, exc)
            return 0

        count = 0
        for _, obj in inspect.getmembers(module):
            if isinstance(obj, (BaseTool, StructuredTool)):
                self.register(obj)
                count += 1
        return count

    # ── 查询 ──────────────────────────────────────────────

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_or_raise(self, name: str) -> BaseTool:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return tool

    def list_tools(self) -> list[dict[str, Any]]:
        """返回所有工具的元信息（用于 /tools/list 接口）"""
        result = []
        for name, t in self._tools.items():
            schema: dict[str, Any] = {}
            if hasattr(t, "args_schema") and t.args_schema is not None:
                try:
                    schema = t.args_schema.model_json_schema()
                except Exception:  # noqa: BLE001
                    pass
            result.append(
                {
                    "name": name,
                    "description": t.description,
                    "args_schema": schema,
                }
            )
        return result

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


# ── 全局单例 ──────────────────────────────────────────────
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
