"""Context assembly.

Builds the final message list sent to the model from its constituent parts. In
Phase 1 that is just the system prompt plus token-budgeted conversation history;
later phases inject retrieved RAG chunks and memory facts here. Keeping this a
single pure function makes it directly unit-testable and eval-targetable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from watari.core.models import ChatMessage, Role
from watari.core.tokens import count_tokens

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _message_tokens(message: ChatMessage) -> int:
    # A small per-message overhead approximates role/formatting tokens.
    return count_tokens(message.content) + 4


def assemble_context(
    history: Sequence[ChatMessage],
    *,
    system_prompt: str | None = None,
    max_context_tokens: int,
    reserved_response_tokens: int = 0,
) -> list[ChatMessage]:
    """Assemble the message list within a token budget.

    The system prompt is always kept. History is included newest-first until the
    remaining budget is exhausted, then re-ordered chronologically. If a single
    most-recent message plus the system prompt still overflows, it is kept anyway
    (the model server will truncate) so the user always gets a response.
    """
    system = ChatMessage(
        role=Role.SYSTEM,
        content=system_prompt if system_prompt is not None else load_system_prompt(),
    )

    budget = max_context_tokens - reserved_response_tokens - _message_tokens(system)

    kept_reversed: list[ChatMessage] = []
    for message in reversed(history):
        cost = _message_tokens(message)
        if cost > budget and kept_reversed:
            break
        budget -= cost
        kept_reversed.append(message)

    return [system, *reversed(kept_reversed)]
