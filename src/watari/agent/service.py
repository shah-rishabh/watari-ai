"""Agent assembly: build the tool registry, permission manager, and loop.

Centralises the wiring so the CLI and API construct an agent the same way. The
tool set is deliberately small (files + tasks, and web_search only when
explicitly enabled) and there is **no shell tool** — a documented decision, see
``docs/adr/002-no-shell-tool.md``.
"""

from __future__ import annotations

from pathlib import Path

from watari.agent.loop import AgentLoop
from watari.agent.models import Risk
from watari.agent.permissions import ConfirmFn, PermissionManager, deny_all
from watari.agent.registry import AnyTool, ToolRegistry
from watari.agent.tools.files import build_file_tools
from watari.agent.tools.tasks import build_task_tools
from watari.agent.tools.web_search import build_web_search_tool
from watari.config import Settings
from watari.core.llm import OpenAICompatibleProvider


def build_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    tools: list[AnyTool] = [
        *build_file_tools(settings),
        *build_task_tools(settings.db_path),
    ]
    if settings.enable_web_search:
        tools.append(build_web_search_tool())
    for tool in tools:
        registry.register(tool)
    return registry


def build_agent(
    provider: OpenAICompatibleProvider,
    settings: Settings,
    *,
    confirm: ConfirmFn = deny_all,
    allowed: set[str] | None = None,
) -> AgentLoop:
    settings.ensure_workspace()
    registry = build_registry(settings)

    all_names = set(registry.names())
    allowed_names = all_names if allowed is None else (allowed & all_names)

    risk_of: dict[str, Risk] = {}
    for name in registry.names():
        tool = registry.get(name)
        if tool is not None:
            risk_of[name] = tool.risk

    audit_path: Path = settings.data_dir / "audit.jsonl"
    permissions = PermissionManager(
        allowed=allowed_names,
        risk_of=risk_of,
        confirm=confirm,
        audit_path=audit_path,
    )
    return AgentLoop(provider, registry, permissions, settings, allowed=allowed_names)
