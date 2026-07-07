"""`watari evals` command group.

Thin wrapper over the harness. `run` executes suites in an isolated eval store,
writes ``results.json`` + a markdown table, and (optionally) gates against
``evals/thresholds.json`` with a non-zero exit on regression.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from watari.config import get_settings
from watari.evals.calibrate import load_calibration, run_calibration
from watari.evals.gate import check_gate, load_thresholds
from watari.evals.harness import REPO_ROOT, SUITES, run_suites
from watari.evals.report import results_to_json, results_to_markdown
from watari.obs.logging import configure_logging

app = typer.Typer(help="Run and report evaluation suites.", no_args_is_help=True)
console = Console()

_THRESHOLDS = REPO_ROOT / "evals" / "thresholds.json"
_CALIBRATION = REPO_ROOT / "evals" / "datasets" / "judge_calibration_v1.jsonl"


@app.command()
def run(
    suite: Annotated[str, typer.Option(help="Suite to run, or 'all'.")] = "all",
    smoke: Annotated[bool, typer.Option(help="Run only the smoke-tagged subset (for CI).")] = False,
    gate: Annotated[
        bool, typer.Option(help="Fail (exit 1) if any metric is below its floor.")
    ] = False,
    out: Annotated[Path, typer.Option(help="Directory for results.json / results.md.")] = Path(
        "eval-results"
    ),
) -> None:
    """Run evaluation suites and write a report."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json=settings.log_json)
    suites: list[str] = list(SUITES) if suite == "all" else [suite]

    results = asyncio.run(run_suites(suites, settings=settings, smoke_only=smoke))

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(results_to_json(results), encoding="utf-8")
    table = results_to_markdown(results)
    (out / "results.md").write_text(table + "\n", encoding="utf-8")
    console.print(table)

    if gate:
        violations = check_gate(results, load_thresholds(_THRESHOLDS))
        if violations:
            console.print("\n[red]Gate failed:[/red]")
            for v in violations:
                console.print(f"  [red]✗[/red] {v}")
            raise typer.Exit(1)
        console.print("\n[green]Gate passed.[/green]")


@app.command()
def calibrate() -> None:
    """Measure judge-vs-human agreement (Cohen's kappa) on the calibration set."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json=settings.log_json)
    cases = load_calibration(_CALIBRATION)
    report = asyncio.run(run_calibration(cases, settings=settings))
    console.print(
        f"Judge calibration ({report.judge_model}, n={report.n}): "
        f"kappa=[cyan]{report.kappa:.3f}[/cyan] "
        f"accuracy=[cyan]{report.accuracy:.3f}[/cyan]"
    )
