"""Tool registry.

A tool is an async function that takes a validated pydantic argument model and
returns a string. The :class:`Tool` wrapper derives the OpenAI-format JSON schema
from that model (so schema and validation never drift) and carries a risk tier
used by the permission policy.

Deliberately tiny: a handful of tools, each with a tight description. Small local
models misfire on large tool sets, so the registry is a feature, not a framework.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from watari.agent.models import Risk

type ToolFn[ArgsT: BaseModel] = Callable[[ArgsT], Awaitable[str]]

# Tools are heterogeneous in their args model but stored and iterated together;
# AnyTool is the erased type used at collection boundaries.
type AnyTool = "Tool[Any]"


class Tool[ArgsT: BaseModel]:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_model: type[ArgsT],
        risk: Risk,
        fn: ToolFn[ArgsT],
    ) -> None:
        self.name = name
        self.description = description
        self.args_model = args_model
        self.risk = risk
        self._fn = fn

    def json_schema(self) -> dict[str, object]:
        """OpenAI-format tool spec, derived from the pydantic args model."""
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    def validate_args(self, raw: dict[str, object]) -> ArgsT:
        return self.args_model.model_validate(raw)

    async def run(self, args: ArgsT) -> str:
        return await self._fn(args)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, AnyTool] = {}

    def register(self, tool: AnyTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AnyTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self, *, allowed: set[str] | None = None) -> list[dict[str, object]]:
        """JSON schemas for the (optionally allowlist-filtered) tools."""
        return [
            t.json_schema() for name, t in self._tools.items() if allowed is None or name in allowed
        ]
