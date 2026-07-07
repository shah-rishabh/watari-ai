"""Web search tool — opt-in and OFF by default.

Enabling network egress contradicts the "your data never leaves your machine"
pitch, so this tool is only registered when ``enable_web_search`` is set, and the
choice is loud in config and logs. The default build does not include it at all;
this module exists so the capability is a deliberate, documented opt-in rather
than a hidden default.

The implementation here is intentionally a stub returning a clear message: wiring
a real search backend is out of scope for the local-first default and would be a
Phase-6 add. What matters for the threat model is that egress is off unless
explicitly enabled.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from watari.agent.models import Risk
from watari.agent.registry import AnyTool, Tool


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=500)


def build_web_search_tool() -> AnyTool:
    async def web_search(args: WebSearchArgs) -> str:
        return (
            "web_search is enabled but no backend is configured. "
            "This local-first build does not ship a network search provider."
        )

    return Tool(
        name="web_search",
        description="Search the web. Sends the query to an external service.",
        args_model=WebSearchArgs,
        risk=Risk.EXECUTE,
        fn=web_search,
    )
