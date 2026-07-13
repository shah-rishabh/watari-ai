"""Text embeddings.

We use `fastembed` with `BAAI/bge-small-en-v1.5`: 384-dim, ONNX runtime, runs on
CPU with no PyTorch in the dependency tree — so it keeps the Docker image small
and behaves identically in GitHub Actions (embeddings never touch the GPU/VRAM
budget). The :class:`Embedder` protocol is the seam so a different backend can be
dropped in without touching the RAG store or retriever.

fastembed is synchronous and CPU-bound; the async wrappers offload to a thread
so they never block the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol

from fastembed import TextEmbedding

from watari.config import Settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts (documents)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


class FastEmbedEmbedder:
    """:class:`Embedder` backed by fastembed's ONNX models."""

    def __init__(self, settings: Settings) -> None:
        self._model = TextEmbedding(model_name=settings.embed_model)
        self.dim = settings.embed_dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # fastembed returns a generator of numpy arrays.
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    async def aembed(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)
