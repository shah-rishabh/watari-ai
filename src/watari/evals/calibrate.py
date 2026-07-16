"""Judge calibration: measure agreement between the LLM judge and human labels.

Runs the judge's per-claim faithfulness check over a hand-labeled set and reports
Cohen's kappa plus raw accuracy. This is what lets us state, with a number, how
much to trust the judge metrics — see ``docs/evals.md``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from watari.config import Settings
from watari.core.llm import OpenAICompatibleProvider
from watari.evals.metrics.calibration import cohens_kappa
from watari.evals.metrics.judge import Judge


class CalibrationCase(BaseModel):
    id: str
    context: str
    answer: str
    human_faithful: bool


class CalibrationReport(BaseModel):
    n: int
    kappa: float
    accuracy: float
    judge_model: str


def load_calibration(path: Path) -> list[CalibrationCase]:
    cases: list[CalibrationCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(CalibrationCase.model_validate_json(line))
    return cases


async def run_calibration(cases: list[CalibrationCase], *, settings: Settings) -> CalibrationReport:
    provider = OpenAICompatibleProvider(settings)
    judge = Judge(provider, settings.judge_model)
    try:
        human: list[bool] = []
        judged: list[bool] = []
        for case in cases:
            supported = await judge.claim_supported(case.answer, case.context)
            human.append(case.human_faithful)
            judged.append(supported)
    finally:
        await provider.aclose()

    n = len(cases)
    accuracy = sum(1 for h, j in zip(human, judged, strict=True) if h == j) / n if n else 0.0
    return CalibrationReport(
        n=n,
        kappa=cohens_kappa(human, judged),
        accuracy=accuracy,
        judge_model=settings.judge_model,
    )
