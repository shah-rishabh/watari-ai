"""Application configuration.

Settings are loaded from environment variables (prefix ``WATARI_``) and an
optional ``.env`` file. Model choice is deliberately expressed as *named roles*
rather than a single global model: the same code runs a 3B chat model on a
laptop GPU and a 0.5B model on CPU in CI simply by overriding these values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    return Path.home() / ".watari"


class Settings(BaseSettings):
    """Runtime configuration for the whole application.

    Every field is overridable via ``WATARI_<FIELD>`` environment variables,
    which is how CI swaps in a tiny CPU model without touching code.
    """

    model_config = SettingsConfigDict(
        env_prefix="WATARI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider (OpenAI-compatible; points at Ollama by default) ---
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible base URL. Ollama exposes /v1 here.",
    )
    llm_api_key: str = Field(
        default="ollama",
        description="Ignored by Ollama but required by the OpenAI client.",
    )
    request_timeout_s: float = Field(default=120.0, ge=1.0)

    # --- Named model roles (all independently overridable) ---
    # Qwen 3.5 4B (Q4) sits around ~4GB VRAM, leaving headroom on a 6GB card for
    # RAG context + KV cache. Its hybrid attention keeps the KV cache small, and
    # it has the most reliable tool-calling / structured-JSON output at this size
    # — which matters for the agent loop, the LLM-judge, and memory extraction.
    # Upgrade to qwen3.5:9b if you have the VRAM/patience for deeper reasoning.
    chat_model: str = Field(default="qwen3.5:4b")
    judge_model: str = Field(default="qwen3.5:4b")
    extract_model: str = Field(default="qwen3.5:4b")
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # --- Generation defaults ---
    # Reasoning ("thinking") models such as Qwen 3.5 emit a long hidden chain of
    # thought before the visible answer, which costs latency and context budget.
    # This maps to the OpenAI-spec `reasoning_effort` parameter: "none" disables
    # thinking (our default, for a snappy interactive assistant), while
    # "low"/"medium"/"high" trade latency for deliberation. Servers that don't
    # support it ignore the field.
    reasoning_effort: str = Field(default="none")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_context_tokens: int = Field(default=8192, ge=512)
    max_response_tokens: int = Field(default=1024, ge=1)

    # --- RAG ---
    # bge-small is 384-dim; keep this in sync if you change embed_model.
    embed_dim: int = Field(default=384, ge=1)
    chunk_target_tokens: int = Field(default=400, ge=32)
    chunk_overlap_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    retrieval_top_k: int = Field(default=5, ge=1)
    # Reciprocal Rank Fusion constant; 60 is the value from the original RRF
    # paper and a robust default.
    rrf_k: int = Field(default=60, ge=1)

    # --- Storage ---
    data_dir: Path = Field(default_factory=_default_data_dir)

    # --- API server ---
    # Bind to loopback by default: a localhost API is reachable by malicious
    # web pages via CSRF-style requests, so we never listen on 0.0.0.0 outside
    # an explicitly configured container.
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)

    # --- Observability ---
    log_level: str = Field(default="INFO")
    log_json: bool = Field(
        default=False,
        description="Emit JSON logs (True) or human-readable console logs (False).",
    )

    @property
    def db_path(self) -> Path:
        """Single SQLite file holding sessions, vectors, and memory."""
        return self.data_dir / "watari.db"

    def ensure_data_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
