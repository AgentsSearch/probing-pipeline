"""Data models for Stage 1: Task Analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubtaskNode:
    """A single subtask in the decomposed query DAG."""

    id: str
    description: str
    capability: str
    difficulty: float
    is_discriminative: bool
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError(f"difficulty must be in [0, 1], got {self.difficulty}")


@dataclass
class TaskDAG:
    """Directed Acyclic Graph representing a decomposed user query."""

    query: str
    intent: str
    domain: str
    nodes: list[SubtaskNode]
    edges: list[tuple[str, str]]
    critical_path: list[str]
    estimated_difficulty: float
    evaluation_dimensions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.estimated_difficulty <= 1.0:
            raise ValueError(
                f"estimated_difficulty must be in [0, 1], got {self.estimated_difficulty}"
            )
        if len(self.nodes) > 6:
            raise ValueError(f"Maximum 6 subtask nodes allowed, got {len(self.nodes)}")

    def get_node(self, node_id: str) -> SubtaskNode | None:
        """Return the node with the given ID, or None."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def discriminative_nodes(self) -> list[SubtaskNode]:
        """Return only nodes marked as discriminative."""
        return [n for n in self.nodes if n.is_discriminative]
