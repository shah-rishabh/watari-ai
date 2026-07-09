"""Watari command-line interface (typer + rich).

The primary demo surface. ``watari chat`` opens an interactive REPL that streams
assistant replies as live Markdown; ``watari serve`` launches the API.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from watari.api.deps import AppState, build_state, teardown_state
from watari.config import get_settings
from watari.obs.logging import configure_logging

app = typer.Typer(
    name="watari",
    help="A local-first LLM personal assistant.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


async def _run_repl() -> None:
    settings = get_settings()
    settings.ensure_data_dir()
    configure_logging(level=settings.log_level, json=settings.log_json)

    state: AppState = await build_state(settings)
    try:
        session_id = await state.store.create_session()
        console.print(
            Panel.fit(
                f"[bold]Watari[/bold] — model [cyan]{settings.chat_model}[/cyan]\n"
                "Type your message. [dim]/exit[/dim] to quit.",
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

            await _stream_turn(state, session_id, user_text)
    finally:
        await teardown_state(state)


async def _stream_turn(state: AppState, session_id: str, user_text: str) -> None:
    buffer = ""
    console.print("[bold magenta]watari ›[/bold magenta]")
    with Live(console=console, refresh_per_second=12, vertical_overflow="visible") as live:
        async for delta in state.chat.stream_reply(session_id, user_text):
            if delta.content:
                buffer += delta.content
                live.update(Markdown(buffer))
    console.print()


@app.command()
def chat() -> None:
    """Start an interactive chat session."""
    asyncio.run(_run_repl())


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
