"""Agent eval suite — tool-selection accuracy and task completion.

Each case gives a task, the tool(s) a correct solution should use, and a
**deterministic completion check** (a file that must exist with certain content,
or a task row that must be created). Completion is asserted against real state,
not judged — the strongest possible signal for an agent eval.

Metrics:
- tool_selection: fraction of cases where the expected tool(s) were actually used
- task_completion: fraction where the deterministic post-condition holds
- mean_iterations: average loop iterations (efficiency signal)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import aiosqlite
from pydantic import BaseModel

from watari.agent.permissions import approve_all
from watari.agent.service import build_agent
from watari.config import Settings
from watari.core.llm import OpenAICompatibleProvider
from watari.core.session import SessionStore
from watari.evals.models import MetricResult, SuiteResult
from watari.security.sandbox import Sandbox

_SYSTEM = (
    "You are Watari, a local assistant with tools. Use them to accomplish the "
    "task, then give a brief final answer. Workspace paths are relative."
)


class AgentCase(BaseModel):
    id: str
    task: str
    expected_tools: list[str] = []
    assert_file: str | None = None
    assert_contains: str | None = None
    assert_task: str | None = None
    tags: list[str] = []


def load_agent(path: Path, *, smoke_only: bool = False) -> list[AgentCase]:
    cases: list[AgentCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = AgentCase.model_validate_json(line)
        if smoke_only and "smoke" not in case.tags:
            continue
        cases.append(case)
    return cases


async def _completed(case: AgentCase, settings: Settings) -> bool:
    if case.assert_file is not None:
        target = Sandbox(settings.workspace_path).resolve(case.assert_file)
        if not target.is_file():
            return False
        if case.assert_contains is not None:
            return (
                case.assert_contains.lower()
                in target.read_text(encoding="utf-8", errors="replace").lower()
            )
        return True
    if case.assert_task is not None:
        async with aiosqlite.connect(settings.db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute("SELECT title FROM tasks")).fetchall()
        needle = case.assert_task.lower()
        return any(needle in r["title"].lower() for r in rows)
    return False


async def run_agent_suite(cases: list[AgentCase], *, settings: Settings) -> SuiteResult:
    # Each case runs in its own isolated data dir so file/task state can't leak
    # between cases and completion checks are unambiguous.
    tool_hits = 0
    completions = 0
    total_iters = 0

    for case in cases:
        with tempfile.TemporaryDirectory(prefix="watari-agent-eval-") as tmp:
            case_settings = settings.model_copy(update={"data_dir": Path(tmp)})
            case_settings.ensure_workspace()
            store = SessionStore(case_settings.db_path)
            await store.connect()
            await store.close()

            provider = OpenAICompatibleProvider(case_settings)
            agent = build_agent(provider, case_settings, confirm=approve_all)
            try:
                outcome = await agent.run(_SYSTEM, case.task)
            finally:
                await provider.aclose()

            used = {c.name for c in outcome.tool_calls}
            if set(case.expected_tools) <= used:
                tool_hits += 1
            total_iters += outcome.iterations
            if await _completed(case, case_settings):
                completions += 1

    n = len(cases) or 1
    metrics = [
        MetricResult(name="tool_selection", value=tool_hits / n, n=len(cases)),
        MetricResult(name="task_completion", value=completions / n, n=len(cases)),
        MetricResult(name="mean_iterations", value=total_iters / n, n=len(cases)),
    ]
    return SuiteResult(
        suite="agent", model=settings.chat_model, n_cases=len(cases), metrics=metrics
    )
