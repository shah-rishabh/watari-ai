"""LLM-judge logic tested with a scripted provider (no live model)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from watari.core.models import ChatDelta, ChatMessage, Usage
from watari.evals.metrics.judge import Judge


class ScriptedProvider:
    """Returns queued responses in order, one per stream() call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatDelta]:
        text = self._responses[self.calls] if self.calls < len(self._responses) else "{}"
        self.calls += 1
        yield ChatDelta(content=text)
        yield ChatDelta(done=True, usage=Usage())


async def test_faithfulness_all_supported() -> None:
    # 1st call: decompose -> 2 claims. Next 2 calls: both supported.
    provider = ScriptedProvider(
        [
            '{"claims": ["Mara lives in Lisbon", "Mara grew up in Enugu"]}',
            '{"supported": "yes"}',
            '{"supported": "yes"}',
        ]
    )
    judge = Judge(provider, model="test")
    result = await judge.faithfulness("answer", "context")
    assert result.n_claims == 2
    assert result.n_supported == 2
    assert result.score == 1.0


async def test_faithfulness_partial() -> None:
    provider = ScriptedProvider(
        [
            '{"claims": ["true claim", "false claim"]}',
            '{"supported": "yes"}',
            '{"supported": "no"}',
        ]
    )
    judge = Judge(provider, model="test")
    result = await judge.faithfulness("answer", "context")
    assert result.score == 0.5


async def test_faithfulness_no_claims_is_one() -> None:
    provider = ScriptedProvider(['{"claims": []}'])
    judge = Judge(provider, model="test")
    result = await judge.faithfulness("answer", "context")
    assert result.score == 1.0


async def test_answer_relevance_scales_to_unit() -> None:
    judge = Judge(ScriptedProvider(['{"score": 2}']), model="test")
    assert await judge.answer_relevance("q", "a") == 1.0

    judge = Judge(ScriptedProvider(['{"score": 1}']), model="test")
    assert await judge.answer_relevance("q", "a") == 0.5

    judge = Judge(ScriptedProvider(['{"score": 0}']), model="test")
    assert await judge.answer_relevance("q", "a") == 0.0


async def test_malformed_json_retries_then_conservative() -> None:
    # Both attempts unparseable -> claim decomposition yields nothing.
    provider = ScriptedProvider(["not json", "still not json"])
    judge = Judge(provider, model="test")
    claims = await judge.decompose_claims("answer")
    assert claims == []
    assert provider.calls == 2  # retried once


async def test_json_embedded_in_prose_is_parsed() -> None:
    provider = ScriptedProvider(['Sure! {"supported": "yes"} hope that helps'])
    judge = Judge(provider, model="test")
    assert await judge.claim_supported("claim", "context") is True
