"""Context assembly is the most logic-heavy pure function in Phase 1."""

from __future__ import annotations

from watari.core.context import assemble_context
from watari.core.models import ChatMessage, Role


def _history(n: int) -> list[ChatMessage]:
    return [
        ChatMessage(
            role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
            content=f"message number {i}",
        )
        for i in range(n)
    ]


def test_system_prompt_is_always_first() -> None:
    result = assemble_context(_history(4), system_prompt="SYS", max_context_tokens=1000)
    assert result[0].role is Role.SYSTEM
    assert result[0].content == "SYS"


def test_full_history_fits_when_budget_is_large() -> None:
    history = _history(6)
    result = assemble_context(history, system_prompt="SYS", max_context_tokens=10_000)
    assert [m.content for m in result[1:]] == [m.content for m in history]


def test_history_is_trimmed_oldest_first_under_tight_budget() -> None:
    history = _history(10)
    # Tiny budget: only the most recent messages survive, but chronological order
    # is preserved in the output.
    result = assemble_context(
        history,
        system_prompt="SYS",
        max_context_tokens=60,
        reserved_response_tokens=0,
    )
    kept = [m.content for m in result[1:]]
    assert kept == sorted(kept, key=lambda c: int(c.split()[-1]))
    assert history[-1].content in kept
    assert history[0].content not in kept


def test_latest_message_kept_even_if_over_budget() -> None:
    long_msg = ChatMessage(role=Role.USER, content="word " * 500)
    result = assemble_context([long_msg], system_prompt="SYS", max_context_tokens=10)
    assert result[-1].content == long_msg.content


def test_reserved_response_tokens_reduce_available_budget() -> None:
    history = _history(10)
    loose = assemble_context(history, system_prompt="SYS", max_context_tokens=200)
    tight = assemble_context(
        history,
        system_prompt="SYS",
        max_context_tokens=200,
        reserved_response_tokens=150,
    )
    assert len(tight) <= len(loose)
