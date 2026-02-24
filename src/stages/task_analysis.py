"""Stage 1: Task Analysis — decompose a user query into a TaskDAG.

Input:  user query (string)
Output: TaskDAG
Cost:   1 LLM call per query (shared across all candidate agents)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.llm.client import LLMClient
from src.models.task import SubtaskNode, TaskDAG

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "prompts" / "task_analysis.txt"

_MAX_NODES = 6


def _load_prompt_template() -> str:
    """Load the task analysis prompt template from disk."""
    with open(_PROMPT_PATH) as f:
        return f.read()


def _parse_task_dag(query: str, data: dict[str, Any]) -> TaskDAG:
    """Convert raw LLM JSON output into a validated TaskDAG.

    Args:
        query: The original user query (used as fallback if missing from response).
        data: Parsed JSON dict from the LLM.

    Returns:
        A validated TaskDAG instance.

    Raises:
        ValueError: If the data is malformed or fails validation.
    """
    nodes_raw = data.get("nodes", [])

    # Enforce max node count by merging lowest-difficulty surplus nodes
    if len(nodes_raw) > _MAX_NODES:
        logger.warning(
            "LLM returned %d nodes, merging to %d", len(nodes_raw), _MAX_NODES
        )
        nodes_raw = _merge_excess_nodes(nodes_raw, _MAX_NODES)

    nodes = [
        SubtaskNode(
            id=n["id"],
            description=n["description"],
            capability=n["capability"],
            difficulty=float(n["difficulty"]),
            is_discriminative=bool(n.get("is_discriminative", False)),
            depends_on=n.get("depends_on", []),
        )
        for n in nodes_raw
    ]

    edges = [tuple(e) for e in data.get("edges", [])]

    return TaskDAG(
        query=data.get("query", query),
        intent=data.get("intent", "unknown"),
        domain=data.get("domain", "unknown"),
        nodes=nodes,
        edges=edges,
        critical_path=data.get("critical_path", []),
        estimated_difficulty=float(data.get("estimated_difficulty", 0.5)),
        evaluation_dimensions=data.get("evaluation_dimensions", []),
    )


def _merge_excess_nodes(
    nodes: list[dict[str, Any]], max_count: int
) -> list[dict[str, Any]]:
    """Merge the least complex nodes until we're at max_count.

    Merges the two lowest-difficulty, non-discriminative nodes into one combined node.
    """
    nodes = sorted(nodes, key=lambda n: (n.get("is_discriminative", False), n.get("difficulty", 0)))

    while len(nodes) > max_count:
        # Take the two easiest non-discriminative nodes
        a = nodes.pop(0)
        b = nodes.pop(0)
        merged = {
            "id": a["id"],
            "description": f"{a['description']}; {b['description']}",
            "capability": a.get("capability", b.get("capability", "general")),
            "difficulty": max(a.get("difficulty", 0), b.get("difficulty", 0)),
            "is_discriminative": a.get("is_discriminative", False) or b.get("is_discriminative", False),
            "depends_on": list(set(a.get("depends_on", []) + b.get("depends_on", []))),
        }
        nodes.insert(0, merged)

    return nodes


def analyse_task(query: str, llm: LLMClient) -> TaskDAG:
    """Decompose a user query into a TaskDAG using the controller LLM.

    Args:
        query: The natural language user query.
        llm: An initialised LLMClient instance.

    Returns:
        A validated TaskDAG representing the decomposed query.

    Raises:
        RuntimeError: If the LLM call fails after retries.
        ValueError: If the LLM output cannot be parsed into a valid TaskDAG.
    """
    template = _load_prompt_template()
    prompt = template.replace("{query}", query)

    logger.info("Running task analysis for query: %s", query[:80])

    data = llm.complete_json(
        prompt,
        system="You are a task decomposition engine. Return only valid JSON.",
    )

    try:
        dag = _parse_task_dag(query, data)
    except (KeyError, TypeError, ValueError) as e:
        logger.error("Failed to parse TaskDAG from LLM output: %s", e)
        raise ValueError(f"Invalid TaskDAG from LLM: {e}") from e

    logger.info(
        "Task analysis complete: %d nodes, critical_path=%s, difficulty=%.2f",
        len(dag.nodes),
        dag.critical_path,
        dag.estimated_difficulty,
    )

    return dag
