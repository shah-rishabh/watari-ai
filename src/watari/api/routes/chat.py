"""Chat endpoints: create a session and stream a reply over SSE."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from watari.api.deps import AppState, get_state
from watari.core.session import new_session_id
from watari.obs.logging import bind_contextvars, clear_contextvars, get_logger

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)

StateDep = Annotated[AppState, Depends(get_state)]


class CreateSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=32_000)


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(state: StateDep) -> CreateSessionResponse:
    session_id = await state.store.create_session()
    return CreateSessionResponse(session_id=session_id)


@router.post("/chat")
async def chat(req: ChatRequest, state: StateDep) -> EventSourceResponse:
    if not await state.store.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="unknown session_id")

    request_id = new_session_id()

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        bind_contextvars(request_id=request_id, session_id=req.session_id)
        try:
            async for delta in state.chat.stream_reply(req.session_id, req.message):
                if delta.content:
                    yield {"event": "delta", "data": delta.content}
                if delta.done:
                    usage = delta.usage.model_dump() if delta.usage else {}
                    yield {"event": "done", "data": json.dumps(usage)}
        except Exception as exc:
            logger.exception("chat.stream_failed", error=str(exc))
            yield {"event": "error", "data": "internal error"}
        finally:
            clear_contextvars()

    return EventSourceResponse(event_stream())
