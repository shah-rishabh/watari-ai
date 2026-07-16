"""Filesystem tools, jailed to the workspace.

read_file / list_dir are READ (auto-approved); write_file is WRITE (requires
confirmation). Every path passes through the :class:`Sandbox`, and reads/writes
are size-capped. File *contents* returned to the model are wrapped as untrusted
data so the model never treats a file it read as instructions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from watari.agent.models import Risk
from watari.agent.registry import AnyTool, Tool
from watari.config import Settings
from watari.security.sandbox import Sandbox
from watari.security.validation import truncate, wrap_untrusted


class ReadFileArgs(BaseModel):
    path: str = Field(description="Workspace-relative path to read.")


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Workspace-relative directory.")


class WriteFileArgs(BaseModel):
    path: str = Field(description="Workspace-relative path to write.")
    content: str = Field(description="Text content to write.")


def build_file_tools(settings: Settings) -> list[AnyTool]:
    sandbox = Sandbox(settings.workspace_path)

    async def read_file(args: ReadFileArgs) -> str:
        target = sandbox.resolve(args.path)
        if not target.is_file():
            return f"error: no such file: {args.path}"
        data = target.read_bytes()[: settings.max_file_read_bytes]
        text = data.decode("utf-8", errors="replace")
        return wrap_untrusted(text, label=args.path)

    async def list_dir(args: ListDirArgs) -> str:
        target = sandbox.resolve(args.path)
        if not target.is_dir():
            return f"error: no such directory: {args.path}"
        entries = sorted(f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir())
        listing = "\n".join(entries) if entries else "(empty)"
        return wrap_untrusted(listing, label=f"listing of {args.path}")

    async def write_file(args: WriteFileArgs) -> str:
        target = sandbox.resolve(args.path)
        payload = args.content.encode("utf-8")
        if len(payload) > settings.max_file_write_bytes:
            return (
                f"error: content exceeds max write size "
                f"({len(payload)} > {settings.max_file_write_bytes} bytes)"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return f"wrote {len(payload)} bytes to {args.path}"

    return [
        Tool(
            name="read_file",
            description="Read a text file from the workspace.",
            args_model=ReadFileArgs,
            risk=Risk.READ,
            fn=read_file,
        ),
        Tool(
            name="list_dir",
            description="List the contents of a workspace directory.",
            args_model=ListDirArgs,
            risk=Risk.READ,
            fn=list_dir,
        ),
        Tool(
            name="write_file",
            description="Write a text file into the workspace. Requires approval.",
            args_model=WriteFileArgs,
            risk=Risk.WRITE,
            fn=write_file,
        ),
    ]


# truncate is imported for callers that want to bound tool output further.
__all__ = ["build_file_tools", "truncate"]
