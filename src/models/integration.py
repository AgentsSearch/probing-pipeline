"""Data models for integration with other streams (B, C, E)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.models.probe import Probe


@dataclass
class RemoteEndpoint:
    """A remote endpoint for an MCP server agent."""

    type: Literal["streamable-http", "sse"]
    url: str


@dataclass
class LLMExtracted:
    """LLM-extracted metadata about an agent (mirrors search API structure)."""

    capabilities: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)


@dataclass
class InlineTool:
    """An inline tool definition provided by the search API."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateAgent:
    """A candidate agent received from Stream B (Retrieval).

    Supports both the legacy 5-field format and the rich schema from the
    search API. Legacy fields ``retrieval_score`` and ``mcp_server_url``
    are kept as property aliases for backward compatibility.
    """

    agent_id: str
    score: float
    remotes: list[RemoteEndpoint] = field(default_factory=list)
    arena_elo: float | None = None
    community_rating: float | None = None
    # Rich schema fields
    description: str | None = None
    tools: list[InlineTool] = field(default_factory=list)
    llm_extracted: LLMExtracted | None = None
    documentation_quality: float | None = None
    is_available: bool = True
    testability_tier: str | None = None

    # ── backward-compat constructor support ──
    def __init__(
        self,
        agent_id: str,
        score: float | None = None,
        remotes: list[RemoteEndpoint] | None = None,
        arena_elo: float | None = None,
        community_rating: float | None = None,
        description: str | None = None,
        tools: list[InlineTool] | None = None,
        llm_extracted: LLMExtracted | None = None,
        documentation_quality: float | None = None,
        is_available: bool = True,
        testability_tier: str | None = None,
        # Deprecated aliases
        retrieval_score: float | None = None,
        mcp_server_url: str | None = None,
    ) -> None:
        self.agent_id = agent_id

        # Handle score / retrieval_score
        if score is not None:
            self.score = score
        elif retrieval_score is not None:
            self.score = retrieval_score
        else:
            raise TypeError("Either 'score' or 'retrieval_score' is required")

        # Handle remotes / mcp_server_url
        if remotes is not None:
            self.remotes = remotes
        elif mcp_server_url is not None:
            self.remotes = [RemoteEndpoint(type="sse", url=mcp_server_url)]
        else:
            self.remotes = []

        self.arena_elo = arena_elo
        self.community_rating = community_rating
        self.description = description
        self.tools = tools or []
        self.llm_extracted = llm_extracted
        self.documentation_quality = documentation_quality
        self.is_available = is_available
        self.testability_tier = testability_tier

    @property
    def retrieval_score(self) -> float:
        """Deprecated alias for ``score``."""
        return self.score

    @property
    def mcp_server_url(self) -> str | None:
        """Deprecated alias — returns ``best_remote_url()``."""
        return self.best_remote_url()

    def best_remote_url(self) -> str | None:
        """Pick the preferred remote endpoint URL.

        Preference order: streamable-http > sse.
        Returns None if no remotes are configured.
        """
        if not self.remotes:
            return None
        # Prefer streamable-http
        for r in self.remotes:
            if r.type == "streamable-http":
                return r.url
        return self.remotes[0].url


@dataclass
class RetrievalResult:
    """Input from Stream B: query + top-k candidate agents."""

    query: str
    candidates: list[CandidateAgent]


@dataclass
class ActionStep:
    """A single step in the controller's action log during probe execution."""

    action: str
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    result: Any = None
    error: str | None = None
    latency_ms: int = 0


@dataclass
class ProbeExecutionRequest:
    """Request sent to Stream C (Sandbox) for probe execution."""

    agent_id: str
    probes: list[Probe]
    total_timeout: int
    remotes: list[RemoteEndpoint] = field(default_factory=list)
    mcp_server_url: str | None = None


@dataclass
class ProbeExecutionResult:
    """Result received from Stream C after probe execution."""

    agent_id: str
    probe_id: str
    output: Any
    trajectory: list[ActionStep]
    latency_ms: int
    success: bool
    error_info: str | None = None


@dataclass
class RankedAgent:
    """Final output sent to Stream E (API/UI)."""

    agent_id: str
    theta: float
    sigma: float
    confidence: float
    testability_tier: str
    probe_summary: str
    prior_influence: float
