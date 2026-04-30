from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=32)
def _load_template(name: str) -> Template:
    path = PROMPTS_DIR / f"{name}.md"
    return Template(path.read_text(encoding="utf-8"))


def render_prompt(name: str, **values: Any) -> str:
    safe_values = {key: str(value) for key, value in values.items()}
    return _load_template(name).safe_substitute(safe_values)
