"""Data models for Stage 2: Tool-Task Alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ParameterMap:
    """Mapping from a subtask parameter to a tool parameter."""

    subtask_param: str
    tool_param: str
    transform: str | None = None


@dataclass
class ToolAlignment:
    """Alignment between a single subtask and a specific tool."""

    subtask_id: str
    tool_name: str
    server_id: str
    match_type: Literal["direct", "partial", "inferred", "none"]
    confidence: float
    retrieval_score: float
    rerank_score: float
    parameter_mapping: dict[str, ParameterMap] = field(default_factory=dict)
    tool_description: str = ""
    tool_parameter_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class AlignmentMap:
    """Full alignment result for one candidate agent."""

    agent_id: str
    server_tool_count: int
    tools_evaluated: int
    alignments: list[ToolAlignment]
    coverage_score: float
    unmatched_subtasks: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage_score <= 1.0:
            raise ValueError(f"coverage_score must be in [0, 1], got {self.coverage_score}")

    def best_alignment_for_subtask(self, subtask_id: str) -> ToolAlignment | None:
        """Return the highest-confidence alignment for a given subtask."""
        matches = [a for a in self.alignments if a.subtask_id == subtask_id]
        if not matches:
            return None
        return max(matches, key=lambda a: a.confidence)
