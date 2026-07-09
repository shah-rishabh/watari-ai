"""Structured logging via structlog.

Every log line carries contextual identifiers (``request_id``, ``session_id``)
bound through structlog's contextvars, so a single request can be traced across
async boundaries without threading a logger argument everywhere.
"""

from __future__ import annotations

import logging

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
from structlog.typing import EventDict, Processor, WrappedLogger

_configured = False


def configure_logging(*, level: str = "INFO", json: bool = False) -> None:
    """Configure structlog + stdlib logging once per process."""
    global _configured
    if _configured:
        return

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _redact_secrets,
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


_SECRET_KEYS = {"api_key", "llm_api_key", "authorization", "token", "password"}


def _redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Redact values whose keys look sensitive before rendering."""
    for key in list(event_dict):
        if key.lower() in _SECRET_KEYS and event_dict[key]:
            event_dict[key] = "***redacted***"
    return event_dict


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


__all__ = [
    "bind_contextvars",
    "clear_contextvars",
    "configure_logging",
    "get_logger",
]
