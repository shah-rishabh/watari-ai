"""Permission policy and audit log for tool execution.

Policy:
- A session-level **allowlist** decides which tools may run at all.
- READ tools are auto-approved.
- WRITE / EXECUTE tools require explicit confirmation. The confirmation is
  supplied by an injectable callback so the CLI can prompt interactively while
  the API can return a pending-approval event.

Every decision and every invocation is appended to a JSONL **audit log** — cheap
and credible provenance for what the agent did on the user's behalf.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from watari.agent.models import Risk, ToolCall
from watari.obs.logging import get_logger

logger = get_logger(__name__)

# Returns True to approve a WRITE/EXECUTE call. Given the tool name and args.
ConfirmFn = Callable[[str, dict[str, object]], Awaitable[bool]]


async def deny_all(_name: str, _args: dict[str, object]) -> bool:
    return False


async def approve_all(_name: str, _args: dict[str, object]) -> bool:
    return True


class PermissionManager:
    def __init__(
        self,
        *,
        allowed: set[str],
        risk_of: dict[str, Risk],
        confirm: ConfirmFn,
        audit_path: Path | None = None,
    ) -> None:
        self._allowed = allowed
        self._risk_of = risk_of
        self._confirm = confirm
        self._audit_path = audit_path

    async def authorize(self, call: ToolCall) -> tuple[bool, str]:
        """Return (allowed, reason). Logs the decision to the audit trail."""
        if call.name not in self._allowed:
            return self._record(call, False, "not in allowlist")

        risk = self._risk_of.get(call.name, Risk.EXECUTE)
        if risk is Risk.READ:
            return self._record(call, True, "read auto-approved")

        approved = await self._confirm(call.name, call.arguments)
        reason = "confirmed" if approved else "denied by user"
        return self._record(call, approved, reason)

    def _record(self, call: ToolCall, allowed: bool, reason: str) -> tuple[bool, str]:
        self._audit(
            action="authorize",
            tool=call.name,
            allowed=allowed,
            reason=reason,
            arguments=call.arguments,
        )
        return allowed, reason

    def audit_execution(self, call: ToolCall, *, is_error: bool) -> None:
        self._audit(
            action="execute",
            tool=call.name,
            is_error=is_error,
            arguments=call.arguments,
        )

    def _audit(self, **fields: object) -> None:
        # Note: use "action" (not "event") — "event" is structlog's reserved
        # positional and passing it as a kwarg collides with the log message.
        record = {"ts": datetime.now(UTC).isoformat(), **fields}
        logger.info("agent.audit", **fields)
        if self._audit_path is not None:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
