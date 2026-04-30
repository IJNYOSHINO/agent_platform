"""
API 请求/响应数据模型（Pydantic v2）
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import base64
import re
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def username_must_be_safe(cls, value: str) -> str:
        value = value.strip()
        if not re.match(r"^[a-zA-Z0-9_-]+$", value):
            raise ValueError("username may only contain letters, numbers, underscores, and hyphens.")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(..., min_length=1, max_length=255, description="Username or email.")
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("identifier")
    @classmethod
    def identifier_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("identifier must not be blank.")
        return value


class AuthUser(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUser


class PlanStep(BaseModel):
    id: str = Field(..., description="步骤唯一标识，如 step1")
    tool: str = Field(..., description="工具名称")
    input: dict[str, Any] = Field(default_factory=dict, description="工具输入参数")
    depends_on: list[str] = Field(default_factory=list, description="依赖的步骤 ID 列表")
    description: str = Field(default="", description="步骤描述（可选）")

    @field_validator("id")
    @classmethod
    def id_must_be_valid(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(f"Step id '{v}' must be a valid identifier.")
        return v


class ExecutionPlan(BaseModel):
    steps: list[PlanStep]

    @field_validator("steps")
    @classmethod
    def steps_must_not_be_empty(cls, v: list[PlanStep]) -> list[PlanStep]:
        if not v:
            raise ValueError("Plan must contain at least one step.")
        # 校验依赖引用合法性
        ids = {s.id for s in v}
        for step in v:
            for dep in step.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on unknown step '{dep}'."
                    )
        return v


class StepResult(BaseModel):
    step_id: str
    tool_name: str
    input: dict[str, Any]
    output: Any = None
    status: str = "pending"
    duration_seconds: float = 0.0
    error: str | None = None


class MediaItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image", "audio", "pdf"] = Field(..., description="媒体类型")
    data: str = Field(..., min_length=1, description="base64 编码的内容")
    media_type: str = Field(
        default="image/jpeg",
        description="MIME 类型，如 image/jpeg, audio/mp3, application/pdf",
    )

    @field_validator("data")
    @classmethod
    def data_must_be_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Media data must be valid base64.") from exc
        return v

    @field_validator("media_type")
    @classmethod
    def media_type_must_look_like_mime(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError("media_type must be a MIME type.")
        return v

    def to_message_part(self) -> dict[str, Any]:
        if self.type == "image":
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{self.media_type};base64,{self.data}"},
            }
        if self.type == "audio":
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": self.data,
                    "format": self.media_type.split("/")[-1],
                },
            }
        return {
            "type": "text",
            "text": f"[PDF Document, base64 encoded]: {self.data[:200]}...",
        }


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str = Field(..., min_length=1, description="自然语言任务指令")
    media: list[MediaItem] = Field(default_factory=list, description="可选的多模态输入")
    task_id: str | None = Field(
        default=None,
        description="可选的现有任务 ID。如果提供，将复用该 ID 关联的短期记忆（会话上下文）。",
    )

    @field_validator("instruction")
    @classmethod
    def instruction_must_not_be_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("instruction must not be blank.")
        return v

    @field_validator("task_id")
    @classmethod
    def task_id_must_not_be_blank(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("task_id must not be blank.")
        return v


class AsyncExecuteRequest(ExecuteRequest):
    """异步执行请求（同步请求的超集，保留扩展空间）"""
    pass


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    plan: list[PlanStep] = Field(default_factory=list)
    steps_result: list[StepResult] = Field(default_factory=list)
    result: str = ""
    error: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list, description="对话历史记录")


class ToolInfo(BaseModel):
    name: str
    description: str
    args_schema: dict[str, Any] = Field(default_factory=dict)


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]
    total: int
