"""Task/reminder tool — a small local to-do list backed by SQLite.

Provides the "calendar-ish" capability from the plan without CalDAV scope creep.
Adding/completing tasks is WRITE (needs approval); listing is READ.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
from pydantic import BaseModel, Field

from watari.agent.models import Risk
from watari.agent.registry import AnyTool, Tool


class AddTaskArgs(BaseModel):
    title: str = Field(min_length=1, max_length=500, description="Task text.")


class ListTasksArgs(BaseModel):
    include_done: bool = Field(default=False, description="Include completed tasks.")


class CompleteTaskArgs(BaseModel):
    id: int = Field(description="Id of the task to mark done.")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_task_tools(db_path: Path) -> list[AnyTool]:
    async def add_task(args: AddTaskArgs) -> str:
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "INSERT INTO tasks (title, done, created_at) VALUES (?, 0, ?)",
                (args.title, _now_iso()),
            )
            await db.commit()
            return f"added task #{cur.lastrowid}: {args.title}"

    async def list_tasks(args: ListTasksArgs) -> str:
        query = "SELECT id, title, done FROM tasks"
        if not args.include_done:
            query += " WHERE done = 0"
        query += " ORDER BY id"
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute(query)).fetchall()
        if not rows:
            return "(no tasks)"
        return "\n".join(f"#{r['id']} [{'x' if r['done'] else ' '}] {r['title']}" for r in rows)

    async def complete_task(args: CompleteTaskArgs) -> str:
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute("UPDATE tasks SET done = 1 WHERE id = ?", (args.id,))
            await db.commit()
            if cur.rowcount == 0:
                return f"error: no task #{args.id}"
            return f"completed task #{args.id}"

    return [
        Tool(
            name="add_task",
            description="Add a task to the local to-do list. Requires approval.",
            args_model=AddTaskArgs,
            risk=Risk.WRITE,
            fn=add_task,
        ),
        Tool(
            name="list_tasks",
            description="List tasks from the local to-do list.",
            args_model=ListTasksArgs,
            risk=Risk.READ,
            fn=list_tasks,
        ),
        Tool(
            name="complete_task",
            description="Mark a task as done. Requires approval.",
            args_model=CompleteTaskArgs,
            risk=Risk.WRITE,
            fn=complete_task,
        ),
    ]
