"""Watari command-line interface (typer + rich).

The primary demo surface. ``watari chat`` opens an interactive REPL that streams
assistant replies as live Markdown; ``watari serve`` launches the API.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from watari.api.deps import AppState, build_state, teardown_state
from watari.config import get_settings
from watari.evals.cli import app as evals_app
from watari.memory.cli import app as memory_app
from watari.obs.logging import configure_logging

app = typer.Typer(
    name="watari",
    help="A local-first LLM personal assistant.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(evals_app, name="evals")
app.add_typer(memory_app, name="memory")
console = Console()


async def _run_repl() -> None:
    settings = get_settings()
    settings.ensure_data_dir()
    configure_logging(level=settings.log_level, json=settings.log_json)

    state: AppState = await build_state(settings)
    try:
        session_id = await state.store.create_session()
        rag_stats = state.rag_store.stats()
        use_rag = rag_stats["chunks"] > 0
        console.print(
            Panel.fit(
                f"[bold]Watari[/bold] — model [cyan]{settings.chat_model}[/cyan]\n"
                f"RAG: [cyan]{'on' if use_rag else 'off'}[/cyan] "
                f"({rag_stats['documents']} docs, {rag_stats['chunks']} chunks)\n"
                "Type your message. [dim]/rag[/dim] toggles retrieval, "
                "[dim]/remember[/dim] saves facts, [dim]/exit[/dim] to quit.",
                border_style="cyan",
            )
        )
        while True:
            try:
                user_text = console.input("[bold green]you ›[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]bye[/dim]")
                break
            if not user_text:
                continue
            if user_text in {"/exit", "/quit"}:
                console.print("[dim]bye[/dim]")
                break
            if user_text == "/rag":
                use_rag = not use_rag
                console.print(f"[dim]RAG {'on' if use_rag else 'off'}[/dim]")
                continue
            if user_text == "/remember":
                history = await state.store.get_history(session_id)
                facts = await state.memory.remember_from_transcript(
                    history, source=f"session:{session_id[:8]}"
                )
                console.print(
                    f"[dim]remembered {len(facts)} fact(s): "
                    f"{', '.join(f.fact for f in facts) or 'none'}[/dim]"
                )
                continue

            await _stream_turn(state, session_id, user_text, use_rag=use_rag)
    finally:
        await teardown_state(state)


async def _stream_turn(state: AppState, session_id: str, user_text: str, *, use_rag: bool) -> None:
    buffer = ""
    console.print("[bold magenta]watari ›[/bold magenta]")
    with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
        async for delta in state.chat.stream_reply(session_id, user_text, use_rag=use_rag):
            if delta.content:
                buffer += delta.content
                live.update(Markdown(buffer))
    console.print()


@app.command()
def chat() -> None:
    """Start an interactive chat session."""
    asyncio.run(_run_repl())


async def _run_agent(task: str, yolo: bool) -> None:
    from watari.agent.permissions import approve_all
    from watari.agent.service import build_agent
    from watari.core.llm import OpenAICompatibleProvider
    from watari.core.session import SessionStore

    settings = get_settings()
    settings.ensure_data_dir()
    settings.ensure_workspace()
    configure_logging(level=settings.log_level, json=settings.log_json)

    # Ensure the schema (tasks table) exists before the agent uses it.
    session_store = SessionStore(settings.db_path)
    await session_store.connect()
    await session_store.close()

    async def confirm(name: str, args: dict[str, object]) -> bool:
        console.print(f"[yellow]⚠ approve[/yellow] [bold]{name}[/bold]({args})? [dim][y/N][/dim]")
        try:
            return console.input("").strip().lower() in {"y", "yes"}
        except (EOFError, KeyboardInterrupt):
            return False

    provider = OpenAICompatibleProvider(settings)
    agent = build_agent(provider, settings, confirm=approve_all if yolo else confirm)
    system_prompt = (
        "You are Watari, a local assistant with tools. Use them to accomplish the "
        "task, then give a brief final answer. Workspace paths are relative."
    )
    if yolo:
        console.print("[red]--yolo: all tool calls auto-approved[/red]")
    try:
        with console.status("Working…"):
            outcome = await agent.run(system_prompt, task)
        if outcome.tool_calls:
            console.print(
                f"[dim]tools used: {', '.join(c.name for c in outcome.tool_calls)} "
                f"({outcome.iterations} steps)[/dim]"
            )
        if outcome.denied:
            console.print(f"[yellow]denied: {', '.join(outcome.denied)}[/yellow]")
        console.print(Markdown(outcome.answer))
    finally:
        await provider.aclose()


@app.command()
def agent(
    task: Annotated[str, typer.Argument(help="What the agent should do.")],
    yolo: Annotated[bool, typer.Option(help="Auto-approve every tool call (demo only).")] = False,
) -> None:
    """Run the tool-using agent on a single task."""
    asyncio.run(_run_agent(task, yolo))


async def _run_ingest(path: Path) -> None:
    settings = get_settings()
    settings.ensure_data_dir()
    configure_logging(level=settings.log_level, json=settings.log_json)
    state = await build_state(settings)
    try:
        with console.status(f"Ingesting {path} …"):
            result = await asyncio.to_thread(state.ingest.ingest_path, path)
        console.print(
            f"[green]Ingested[/green] {result.ingested_files} file(s), "
            f"skipped {result.skipped_files}, "
            f"{result.total_chunks} new chunk(s)."
        )
    finally:
        await teardown_state(state)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(exists=True, help="File or directory to ingest.")],
) -> None:
    """Ingest personal documents (markdown, PDF) into the local RAG store."""
    asyncio.run(_run_ingest(path))


async def _run_stats() -> None:
    settings = get_settings()
    settings.ensure_data_dir()
    state = await build_state(settings)
    try:
        s = state.rag_store.stats()
        mem = len(state.memory.list_active())
        console.print(
            Panel.fit(
                f"documents: [cyan]{s['documents']}[/cyan]\n"
                f"chunks:    [cyan]{s['chunks']}[/cyan]\n"
                f"memories:  [cyan]{mem}[/cyan]\n"
                f"embed:     [cyan]{settings.embed_model}[/cyan]",
                title="Watari store",
                border_style="cyan",
            )
        )
    finally:
        await teardown_state(state)


@app.command()
def stats() -> None:
    """Show store statistics (documents, chunks, memories)."""
    asyncio.run(_run_stats())


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Override bind host."),
    port: int | None = typer.Option(None, help="Override bind port."),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    settings = get_settings()
    settings.ensure_data_dir()
    uvicorn.run(
        "watari.api.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
    )


if __name__ == "__main__":
    app()
