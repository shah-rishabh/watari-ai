"""API surface: health, session creation, and SSE chat streaming.

Uses a real app instance but swaps the provider construction for a fake so no
model server is required.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import watari.api.app as api_app
import watari.api.deps as api_deps
from tests.conftest import FakeProvider
from watari.api.app import create_app
from watari.api.deps import AppState
from watari.config import Settings


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    settings = Settings(data_dir=tmp_path)

    # Build state with a fake provider instead of the OpenAI client.
    real_build = api_deps.build_state

    async def build_with_fake(s: Settings) -> AppState:
        state = await real_build(s)
        state.chat._provider = FakeProvider(reply="hi from fake")  # type: ignore[assignment]
        return state

    monkeypatch.setattr(api_app, "build_state", build_with_fake)

    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_flow_streams_sse(client: TestClient) -> None:
    session_id = client.post("/sessions").json()["session_id"]
    resp = client.post("/chat", json={"session_id": session_id, "message": "hello"})
    assert resp.status_code == 200
    body = resp.text
    assert "event: delta" in body
    assert "event: done" in body
    assert "fake" in body


def test_chat_unknown_session_404(client: TestClient) -> None:
    resp = client.post("/chat", json={"session_id": "nope", "message": "hi"})
    assert resp.status_code == 404


def test_chat_rejects_empty_message(client: TestClient) -> None:
    session_id = client.post("/sessions").json()["session_id"]
    resp = client.post("/chat", json={"session_id": session_id, "message": ""})
    assert resp.status_code == 422
