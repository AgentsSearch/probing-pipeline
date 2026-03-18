"""Pydantic v2 request/response models for the probing pipeline API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.integration import (
    CandidateAgent,
    InlineTool,
    LLMExtracted,
    RemoteEndpoint,
)


# ── Request models ──


class RemoteEndpointIn(BaseModel):
    type: str
    url: str


class InlineToolIn(BaseModel):
    name: str
    description: str
    input_schema: dict = Field(default_factory=dict)


class LLMExtractedIn(BaseModel):
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)


class CandidateAgentIn(BaseModel):
    agent_id: str
    score: float = Field(ge=0.0, le=1.0)
    remotes: list[RemoteEndpointIn] = Field(default_factory=list)
    arena_elo: float | None = None
    community_rating: float | None = None
    description: str | None = None
    tools: list[InlineToolIn] = Field(default_factory=list)
    llm_extracted: LLMExtractedIn | None = None
    documentation_quality: float | None = None
    is_available: bool = True
    testability_tier: str | None = None


class ProbeRequest(BaseModel):
    query: str = Field(min_length=1)
    candidates: list[CandidateAgentIn] = Field(min_length=1)


# ── Response models ──


class RankedAgentOut(BaseModel):
    agent_id: str
    theta: float
    sigma: float
    confidence: float
    testability_tier: str
    probe_summary: str
    prior_influence: float


class SubtaskOut(BaseModel):
    id: str
    description: str
    capability: str
    difficulty: float
    is_discriminative: bool


class TaskDAGOut(BaseModel):
    query: str
    intent: str
    domain: str
    nodes: list[SubtaskOut]
    critical_path: list[str]
    estimated_difficulty: float


class AgentDetailOut(BaseModel):
    agent_id: str
    ranked: RankedAgentOut | None = None
    error_code: str | None = None
    error_detail: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    probes_generated: int = 0
    probes_validated: int = 0
    probes_executed: int = 0


class TokenUsage(BaseModel):
    input: int
    output: int
    total: int


class ProbeResponse(BaseModel):
    rankings: list[RankedAgentOut]
    task_dag: TaskDAGOut | None = None
    agent_details: list[AgentDetailOut]
    total_time_ms: int
    llm_calls: int
    token_usage: TokenUsage


class ErrorOut(BaseModel):
    error: str
    detail: str


# ── Converters ──


def to_candidate_agent(agent_in: CandidateAgentIn) -> CandidateAgent:
    """Convert a Pydantic CandidateAgentIn to a dataclass CandidateAgent."""
    remotes = [
        RemoteEndpoint(type=r.type, url=r.url) for r in agent_in.remotes
    ]
    tools = [
        InlineTool(name=t.name, description=t.description, input_schema=t.input_schema)
        for t in agent_in.tools
    ]
    llm_extracted = (
        LLMExtracted(
            capabilities=agent_in.llm_extracted.capabilities,
            limitations=agent_in.llm_extracted.limitations,
            requirements=agent_in.llm_extracted.requirements,
        )
        if agent_in.llm_extracted
        else None
    )
    return CandidateAgent(
        agent_id=agent_in.agent_id,
        score=agent_in.score,
        remotes=remotes,
        arena_elo=agent_in.arena_elo,
        community_rating=agent_in.community_rating,
        description=agent_in.description,
        tools=tools,
        llm_extracted=llm_extracted,
        documentation_quality=agent_in.documentation_quality,
        is_available=agent_in.is_available,
        testability_tier=agent_in.testability_tier,
    )
