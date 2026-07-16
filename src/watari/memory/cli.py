"""`watari memory` command group: list / remember / forget / wipe."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from watari.api.deps import AppState, build_state, teardown_state
from watari.config import get_settings
from watari.memory.models import Fact, StoredMemory
from watari.obs.logging import configure_logging

app = typer.Typer(help="Manage long-term memory.", no_args_is_help=True)
console = Console()


def _run[T](action: Callable[[AppState], Awaitable[T]]) -> T:
    async def go() -> T:
        settings = get_settings()
        settings.ensure_data_dir()
        configure_logging(level=settings.log_level, json=settings.log_json)
        state = await build_state(settings)
        try:
            return await action(state)
        finally:
            await teardown_state(state)

    return asyncio.run(go())


@app.command(name="list")
def list_memories() -> None:
    """List everything Watari remembers about you."""

    async def action(state: AppState) -> list[StoredMemory]:
        return state.memory.list_active()

    memories = _run(action)
    if not memories:
        console.print("[dim]No memories stored.[/dim]")
        return
    table = Table(title="Memories")
    table.add_column("id", justify="right")
    table.add_column("category")
    table.add_column("fact")
    for m in memories:
        table.add_row(str(m.id), m.category.value, m.fact)
    console.print(table)


@app.command()
def remember(
    fact: Annotated[str, typer.Argument(help="A fact to remember about you.")],
) -> None:
    """Explicitly store a fact."""

    async def action(state: AppState) -> int:
        return await state.memory.remember_fact(Fact(fact=fact), source="manual")

    mem_id = _run(action)
    console.print(f"[green]Remembered[/green] as memory #{mem_id}.")


@app.command()
def forget(
    memory_id: Annotated[int, typer.Argument(help="Id of the memory to forget.")],
) -> None:
    """Forget a specific memory."""

    async def action(state: AppState) -> bool:
        return state.memory.forget(memory_id)

    if _run(action):
        console.print(f"[green]Forgot[/green] memory #{memory_id}.")
    else:
        console.print(f"[yellow]No active memory #{memory_id}.[/yellow]")


@app.command()
def wipe(
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation.")] = False,
) -> None:
    """Forget everything."""
    if not yes:
        confirmed = console.input("Wipe ALL memories? [y/N] ").strip().lower()
        if confirmed not in {"y", "yes"}:
            console.print("[dim]cancelled[/dim]")
            return

    async def action(state: AppState) -> int:
        return state.memory.wipe()

    console.print(f"[green]Wiped[/green] {_run(action)} memories.")
