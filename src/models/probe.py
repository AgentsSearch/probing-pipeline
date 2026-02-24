"""Data models for Stage 3: Probe Plan Generation and Stage 4: Validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class RubricDimension:
    """A single evaluation dimension within a probe rubric."""

    name: str
    weight: float
    criteria: str
    pass_threshold: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")


@dataclass
class Probe:
    """A concrete, executable test case for an MCP tool."""

    probe_id: str
    targets_subtask: str
    tool: str
    arguments: dict[str, Any]
    estimated_difficulty: float
    discrimination: float
    rubric: list[RubricDimension]
    timeout_seconds: int
    priority: Literal["PRIMARY", "SECONDARY"]

    def __post_init__(self) -> None:
        if not 0.0 <= self.estimated_difficulty <= 1.0:
            raise ValueError(
                f"estimated_difficulty must be in [0, 1], got {self.estimated_difficulty}"
            )
        if len(self.rubric) < 2:
            raise ValueError(f"Rubric must have at least 2 dimensions, got {len(self.rubric)}")


@dataclass
class ProbePlan:
    """A set of probes to execute against a single agent."""

    query: str
    agent_id: str
    strategy: str
    probes: list[Probe]
    total_budget_seconds: int


@dataclass
class ProbeTemplate:
    """Pre-computed probe template for a known MCP tool."""

    template_id: str
    server_id: str
    tool_name: str
    difficulty: float
    discrimination: float
    arg_template: dict[str, Any]
    expected_behaviour: str
    rubric_template: list[RubricDimension]
    validated: bool = False
    created_at: datetime = field(default_factory=datetime.now)
