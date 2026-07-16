"""The tool-use agent loop.

One turn of the loop:
1. Ask the model for a completion, offering the allowlisted tool specs.
2. If it returns a final answer (no tool calls), stop and return it.
3. Otherwise, for each requested call: authorize (permission policy), execute
   the tool if approved (or record a denial), and append the result as a tool
   message.
4. Repeat, up to ``max_iterations``.

Hand-rolled deliberately — the loop is core agent competency, and owning it keeps
the permission checkpoint and untrusted-result wrapping on one obvious path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from watari.agent.models import ToolCall, ToolResult
from watari.agent.permissions import PermissionManager
from watari.agent.registry import ToolRegistry
from watari.config import Settings
from watari.core.llm import OpenAICompatibleProvider
from watari.obs.logging import get_logger
from watari.security.validation import truncate, wrap_untrusted

logger = get_logger(__name__)

_TOOL_RESULT_CHAR_CAP = 8_000


@dataclass
class AgentOutcome:
    answer: str
    iterations: int
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    denied: list[str] = field(default_factory=list[str])


class AgentLoop:
    def __init__(
        self,
        provider: OpenAICompatibleProvider,
        registry: ToolRegistry,
        permissions: PermissionManager,
        settings: Settings,
        *,
        allowed: set[str],
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._permissions = permissions
        self._settings = settings
        self._allowed = allowed

    async def run(self, system_prompt: str, user_text: str) -> AgentOutcome:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        specs = self._registry.specs(allowed=self._allowed)

        executed: list[ToolCall] = []
        denied: list[str] = []

        for i in range(1, self._settings.max_agent_iterations + 1):
            turn = await self._provider.complete_with_tools(messages, specs)

            if not turn.tool_calls:
                return AgentOutcome(
                    answer=turn.content,
                    iterations=i,
                    tool_calls=executed,
                    denied=denied,
                )

            # Record the assistant's tool-call turn so the follow-up tool
            # messages have something to answer to.
            messages.append(self._assistant_tool_message(turn.tool_calls))

            for call in turn.tool_calls:
                result = await self._handle_call(call)
                if result.is_error and "denied" in result.content:
                    denied.append(call.name)
                else:
                    executed.append(call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.content,
                    }
                )

        # Iteration cap hit without a final answer.
        logger.warning("agent.max_iterations", cap=self._settings.max_agent_iterations)
        return AgentOutcome(
            answer="I couldn't complete that within the step limit.",
            iterations=self._settings.max_agent_iterations,
            tool_calls=executed,
            denied=denied,
        )

    async def _handle_call(self, call: ToolCall) -> ToolResult:
        allowed, reason = await self._permissions.authorize(call)
        if not allowed:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"tool call denied: {reason}",
                is_error=True,
            )

        tool = self._registry.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"unknown tool: {call.name}",
                is_error=True,
            )

        try:
            args = tool.validate_args(call.arguments)
            raw = await tool.run(args)
            content = truncate(raw, _TOOL_RESULT_CHAR_CAP)
            self._permissions.audit_execution(call, is_error=False)
            return ToolResult(call_id=call.id, name=call.name, content=content)
        except Exception as exc:  # surfaced back to the model, never raised
            self._permissions.audit_execution(call, is_error=True)
            # Wrap the error as untrusted too: a tool error message can contain
            # attacker-influenced content (e.g. a filename).
            safe = wrap_untrusted(str(exc), label=f"{call.name} error")
            return ToolResult(call_id=call.id, name=call.name, content=safe, is_error=True)

    @staticmethod
    def _assistant_tool_message(calls: list[ToolCall]) -> dict[str, object]:
        import json

        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in calls
            ],
        }
