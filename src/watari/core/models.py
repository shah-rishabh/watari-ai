"""Core domain models shared across every layer.

These are the *only* types that cross the provider seam — no ``openai`` SDK
types leak into ``rag``/``agent``/``memory`` code, which keeps those layers
testable against a mock provider.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str


class Usage(BaseModel):
    """Token accounting for one completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatDelta(BaseModel):
    """One streamed chunk from the model.

    A stream is a sequence of deltas; the final delta carries ``usage`` and
    ``done=True``. Intermediate deltas carry incremental ``content``. Reasoning
    ("thinking") models additionally emit ``reasoning`` deltas — the hidden
    chain of thought — which we surface separately so callers can show or ignore
    it, but which is never persisted as part of the answer.
    """

    content: str = ""
    reasoning: str = ""
    done: bool = False
    usage: Usage | None = Field(default=None)
