"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from watari import __version__
from watari.api.deps import build_state, teardown_state
from watari.api.routes import chat, health
from watari.config import Settings, get_settings
from watari.obs.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, json=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.watari = await build_state(settings)
        try:
            yield
        finally:
            await teardown_state(app.state.watari)

    app = FastAPI(
        title="Watari AI",
        version=__version__,
        summary="A local-first LLM personal assistant.",
        lifespan=lifespan,
    )

    # This API is single-user and localhost-only. We do not allow cross-origin
    # requests by default so a malicious web page cannot drive it via the
    # browser. Same-origin (the bundled UI) is unaffected.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    return app
