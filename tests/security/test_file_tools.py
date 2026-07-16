"""File tools enforce the jail and size caps through the tool boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from watari.agent.tools.files import (
    ReadFileArgs,
    WriteFileArgs,
    build_file_tools,
)
from watari.config import Settings
from watari.security.sandbox import SandboxError


@pytest.fixture
def tools(tmp_path: Path) -> dict[str, object]:
    settings = Settings(
        data_dir=tmp_path,
        workspace_dir=tmp_path / "ws",
        max_file_write_bytes=50,
    )
    settings.ensure_workspace()
    return {t.name: t for t in build_file_tools(settings)}


async def test_write_then_read_roundtrip(tools: dict[str, object]) -> None:
    write = tools["write_file"]
    read = tools["read_file"]
    await write.run(WriteFileArgs(path="note.txt", content="hello world"))  # type: ignore[attr-defined]
    out = await read.run(ReadFileArgs(path="note.txt"))  # type: ignore[attr-defined]
    assert "hello world" in out
    # Read content is wrapped as untrusted.
    assert "UNTRUSTED" in out


async def test_write_over_size_cap_is_refused(tools: dict[str, object]) -> None:
    write = tools["write_file"]
    result = await write.run(WriteFileArgs(path="big.txt", content="x" * 100))  # type: ignore[attr-defined]
    assert "exceeds max write size" in result


async def test_write_outside_jail_raises(tools: dict[str, object]) -> None:
    write = tools["write_file"]
    with pytest.raises(SandboxError):
        await write.run(WriteFileArgs(path="../escape.txt", content="x"))  # type: ignore[attr-defined]


async def test_read_missing_file_returns_error_not_exception(
    tools: dict[str, object],
) -> None:
    read = tools["read_file"]
    out = await read.run(ReadFileArgs(path="nope.txt"))  # type: ignore[attr-defined]
    assert "no such file" in out
