"""Approximate token counting.

We use tiktoken's ``cl100k_base`` as a model-agnostic proxy. It is not the exact
tokenizer for local models (Qwen etc. differ), but for *budgeting* — deciding
how much history fits in a context window — a consistent approximation is all we
need, and it avoids shipping per-model tokenizers.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken


@lru_cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def encode(text: str) -> list[int]:
    return _encoding().encode(text)


def decode(token_ids: list[int]) -> str:
    return _encoding().decode(token_ids)
