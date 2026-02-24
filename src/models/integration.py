"""Data models for integration with other streams (B, C, E)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models.probe import Probe


@dataclass
class CandidateAgent:
    """A candidate agent received from Stream B (Retrieval)."""

    agent_id: str
    retrieval_score: float
    mcp_server_url: str
    arena_elo: float | None = None
    community_rating: float | None = None


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
    mcp_server_url: str
    probes: list[Probe]
    total_timeout: int


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
