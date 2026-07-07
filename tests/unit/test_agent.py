"""Agent registry, permissions, and loop logic (no live model)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from watari.agent.loop import AgentLoop
from watari.agent.models import AssistantTurn, Risk, ToolCall
from watari.agent.permissions import (
    ConfirmFn,
    PermissionManager,
    approve_all,
    deny_all,
)
from watari.agent.registry import Tool, ToolRegistry
from watari.config import Settings


class EchoArgs(BaseModel):
    text: str


def _echo_tool(risk: Risk = Risk.READ, name: str = "echo") -> Tool[EchoArgs]:
    async def echo(args: EchoArgs) -> str:
        return f"echo: {args.text}"

    return Tool(
        name=name,
        description="echo the input",
        args_model=EchoArgs,
        risk=risk,
        fn=echo,
    )


class TestRegistry:
    def test_schema_derived_from_args_model(self) -> None:
        spec = _echo_tool().json_schema()
        fn = spec["function"]
        assert isinstance(fn, dict)
        assert fn["name"] == "echo"
        params = fn["parameters"]
        assert isinstance(params, dict)
        assert "text" in params["properties"]  # type: ignore[index]

    def test_duplicate_registration_raises(self) -> None:
        reg = ToolRegistry()
        reg.register(_echo_tool())
        with pytest.raises(ValueError, match="duplicate"):
            reg.register(_echo_tool())

    def test_specs_respects_allowlist(self) -> None:
        reg = ToolRegistry()
        reg.register(_echo_tool(name="a"))
        reg.register(_echo_tool(name="b"))
        specs = reg.specs(allowed={"a"})
        names = [s["function"]["name"] for s in specs]  # type: ignore[index]
        assert names == ["a"]


class TestPermissions:
    async def test_read_is_auto_approved(self, tmp_path: Path) -> None:
        pm = PermissionManager(
            allowed={"echo"},
            risk_of={"echo": Risk.READ},
            confirm=deny_all,
            audit_path=tmp_path / "audit.jsonl",
        )
        allowed, reason = await pm.authorize(ToolCall(id="1", name="echo", arguments={}))
        assert allowed
        assert "auto-approved" in reason

    async def test_write_requires_confirmation(self, tmp_path: Path) -> None:
        pm = PermissionManager(
            allowed={"w"},
            risk_of={"w": Risk.WRITE},
            confirm=deny_all,
            audit_path=tmp_path / "audit.jsonl",
        )
        allowed, _ = await pm.authorize(ToolCall(id="1", name="w", arguments={}))
        assert not allowed

    async def test_not_in_allowlist_is_denied(self, tmp_path: Path) -> None:
        pm = PermissionManager(
            allowed=set(),
            risk_of={},
            confirm=approve_all,
            audit_path=tmp_path / "audit.jsonl",
        )
        allowed, reason = await pm.authorize(ToolCall(id="1", name="x", arguments={}))
        assert not allowed
        assert "allowlist" in reason

    async def test_audit_log_is_written(self, tmp_path: Path) -> None:
        audit = tmp_path / "audit.jsonl"
        pm = PermissionManager(
            allowed={"echo"},
            risk_of={"echo": Risk.READ},
            confirm=deny_all,
            audit_path=audit,
        )
        await pm.authorize(ToolCall(id="1", name="echo", arguments={"text": "hi"}))
        assert audit.exists()
        assert "authorize" in audit.read_text(encoding="utf-8")


class ScriptedToolProvider:
    """Provider that returns queued AssistantTurns for complete_with_tools."""

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self._turns = turns
        self.calls = 0

    async def complete_with_tools(
        self,
        wire_messages: list[dict[str, object]],
        tools: list[dict[str, object]],
        *,
        model: str | None = None,
    ) -> AssistantTurn:
        turn = self._turns[self.calls]
        self.calls += 1
        return turn


def _loop(
    provider: object,
    tmp_path: Path,
    *,
    confirm: ConfirmFn = approve_all,
    allowed: set[str] | None = None,
) -> AgentLoop:
    reg = ToolRegistry()
    reg.register(_echo_tool(Risk.READ, "echo"))
    reg.register(_echo_tool(Risk.WRITE, "danger"))
    allow = {"echo", "danger"} if allowed is None else allowed
    pm = PermissionManager(
        allowed=allow,
        risk_of={"echo": Risk.READ, "danger": Risk.WRITE},
        confirm=confirm,
        audit_path=tmp_path / "audit.jsonl",
    )
    settings = Settings(data_dir=tmp_path)
    return AgentLoop(provider, reg, pm, settings, allowed=allow)  # type: ignore[arg-type]


class TestLoop:
    async def test_final_answer_without_tools_returns_immediately(self, tmp_path: Path) -> None:
        provider = ScriptedToolProvider([AssistantTurn(content="done")])
        loop = _loop(provider, tmp_path)
        outcome = await loop.run("sys", "hi")
        assert outcome.answer == "done"
        assert outcome.iterations == 1
        assert outcome.tool_calls == []

    async def test_tool_call_then_answer(self, tmp_path: Path) -> None:
        provider = ScriptedToolProvider(
            [
                AssistantTurn(tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})]),
                AssistantTurn(content="the echo said hi"),
            ]
        )
        loop = _loop(provider, tmp_path)
        outcome = await loop.run("sys", "echo hi")
        assert "hi" in outcome.answer
        assert [c.name for c in outcome.tool_calls] == ["echo"]

    async def test_denied_write_is_recorded(self, tmp_path: Path) -> None:
        provider = ScriptedToolProvider(
            [
                AssistantTurn(
                    tool_calls=[ToolCall(id="1", name="danger", arguments={"text": "x"})]
                ),
                AssistantTurn(content="couldn't do it"),
            ]
        )
        loop = _loop(provider, tmp_path, confirm=deny_all)
        outcome = await loop.run("sys", "do danger")
        assert "danger" in outcome.denied

    async def test_iteration_cap_is_enforced(self, tmp_path: Path) -> None:
        # Always returns a tool call, never a final answer -> must hit the cap.
        always_tool = AssistantTurn(
            tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "x"})]
        )
        provider = ScriptedToolProvider([always_tool] * 20)
        loop = _loop(provider, tmp_path)
        outcome = await loop.run("sys", "loop forever")
        settings = Settings(data_dir=tmp_path)
        assert outcome.iterations == settings.max_agent_iterations
