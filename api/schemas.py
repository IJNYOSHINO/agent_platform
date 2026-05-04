"""
API 请求/响应数据模型（Pydantic v2）
"""
from __future__ import annotations

import base64
import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# 从验证模块导入（需要新建）
from api.media_validator import (
    SUPPORTED_IMAGE_MIMES,
    SUPPORTED_AUDIO_MIMES,
    SUPPORTED_PDF_MIMES,
    MediaValidator,
    MediaSanitizer,
    MediaConverter,
)


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
        ids = {s.id for s in v}
        for step in v:
            for dep in step.depends_on:
                if dep not in ids:
                    raise ValueError(f"Step '{step.id}' depends on unknown step '{dep}'.")
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
    """媒体项（纯数据模型）"""
    model_config = ConfigDict(extra="forbid")

    type: Literal["image", "audio", "pdf"] = Field(..., description="媒体类型")
    data: str = Field(..., min_length=1, description="base64 编码的内容")
    media_type: str = Field(
        default="image/jpeg",
        description="MIME 类型，如 image/jpeg, audio/mp3, application/pdf",
    )

    @field_validator("data")
    @classmethod
    def validate_and_sanitize(cls, v: str, info) -> str:
        """验证并处理媒体文件"""
        media_type = info.data.get("type")

        # 1. 验证 Base64 格式
        file_bytes = MediaValidator.validate_base64(v)

        # 2. 验证文件大小
        MediaValidator.validate_file_size(file_bytes)

        # 3. 检测真实 MIME
        real_mime = MediaValidator.detect_mime(file_bytes)

        # 4. 根据类型处理
        if media_type == "image":
            # 图片：无害化处理
            return MediaSanitizer.sanitize_image(file_bytes)
        elif media_type == "audio":
            # 音频：验证格式
            if real_mime:
                MediaValidator.validate_mime("audio", real_mime)
            return base64.b64encode(file_bytes).decode()
        elif media_type == "pdf":
            # PDF：验证格式
            MediaValidator.validate_pdf(file_bytes)
            if real_mime:
                MediaValidator.validate_mime("pdf", real_mime)
            return base64.b64encode(file_bytes).decode()

        return v

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, v: str, info) -> str:
        """验证 media_type 是否与 type 匹配"""
        if "/" not in v:
            raise ValueError("media_type must be a MIME type.")

        media_type = info.data.get("type")

        if media_type == "image" and v not in SUPPORTED_IMAGE_MIMES:
            raise ValueError(f"Unsupported image MIME: {v}")
        if media_type == "audio" and v not in SUPPORTED_AUDIO_MIMES:
            raise ValueError(f"Unsupported audio MIME: {v}")
        if media_type == "pdf" and v not in SUPPORTED_PDF_MIMES:
            raise ValueError(f"Unsupported PDF MIME: {v}")

        return v

    def to_message_part(self) -> dict[str, Any]:
        """转换为 LangChain 消息格式"""
        if self.type == "image":
            return MediaConverter.to_langchain_image(self.data)
        if self.type == "audio":
            return MediaConverter.to_langchain_audio(self.data, self.media_type)
        return MediaConverter.to_langchain_pdf(self.data)


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