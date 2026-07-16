"""Agent/tool domain models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Risk(StrEnum):
    """A tool's risk tier, which drives the permission policy.

    READ is auto-approved; WRITE and EXECUTE require confirmation.
    """

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


class ToolCall(BaseModel):
    """A model-requested tool invocation."""

    id: str
    name: str
    arguments: dict[str, object]


class ToolResult(BaseModel):
    """The outcome of executing a tool call."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


class AssistantTurn(BaseModel):
    """One assistant turn from a tool-enabled completion.

    Either it produced ``content`` (a final answer) or it requested
    ``tool_calls`` (or both, though small models usually do one or the other).
    """

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
