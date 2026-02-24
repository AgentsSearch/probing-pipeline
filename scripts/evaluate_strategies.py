#!/usr/bin/env python3
"""Evaluate and compare probe selection strategies.

Runs different strategies against fixture data and compares their
information gain and discrimination power.

Usage:
    python scripts/evaluate_strategies.py \
        --fixtures tests/fixtures/sample_mcp_servers.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.alignment import AlignmentMap, ToolAlignment
from src.models.probe import Probe, RubricDimension
from src.models.scoring import GaussianPrior
from src.models.task import SubtaskNode, TaskDAG
from src.scoring.birt import bayesian_update, score_agent
from src.stages.probe_generation import _select_subtasks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _make_rubric() -> list[RubricDimension]:
    return [
        RubricDimension(name="correctness", weight=0.6, criteria="c", pass_threshold="p"),
        RubricDimension(name="completeness", weight=0.4, criteria="c", pass_threshold="p"),
    ]


def _synthetic_dag() -> TaskDAG:
    """Create a synthetic TaskDAG for strategy evaluation."""
    nodes = [
        SubtaskNode(id="S1", description="Easy lookup", capability="api_call",
                     difficulty=0.2, is_discriminative=False),
        SubtaskNode(id="S2", description="Core processing", capability="transform",
                     difficulty=0.6, is_discriminative=True, depends_on=["S1"]),
        SubtaskNode(id="S3", description="Complex analysis", capability="analysis",
                     difficulty=0.8, is_discriminative=True, depends_on=["S1"]),
        SubtaskNode(id="S4", description="Format output", capability="formatting",
                     difficulty=0.2, is_discriminative=False, depends_on=["S2", "S3"]),
    ]
    return TaskDAG(
        query="Synthetic evaluation query",
        intent="evaluation",
        domain="test",
        nodes=nodes,
        edges=[("S1", "S2"), ("S1", "S3"), ("S2", "S4"), ("S3", "S4")],
        critical_path=["S1", "S3", "S4"],
        estimated_difficulty=0.55,
    )


def _synthetic_alignment(server_id: str) -> AlignmentMap:
    """Create a synthetic AlignmentMap for evaluation."""
    alignments = [
        ToolAlignment(subtask_id="S1", tool_name="lookup_tool", server_id=server_id,
                       match_type="direct", confidence=0.9, retrieval_score=0.9, rerank_score=0.9),
        ToolAlignment(subtask_id="S2", tool_name="process_tool", server_id=server_id,
                       match_type="direct", confidence=0.85, retrieval_score=0.8, rerank_score=0.85),
        ToolAlignment(subtask_id="S3", tool_name="analyse_tool", server_id=server_id,
                       match_type="partial", confidence=0.7, retrieval_score=0.7, rerank_score=0.7),
        ToolAlignment(subtask_id="S4", tool_name="format_tool", server_id=server_id,
                       match_type="direct", confidence=0.95, retrieval_score=0.95, rerank_score=0.95),
    ]
    return AlignmentMap(
        agent_id=server_id, server_tool_count=4, tools_evaluated=4,
        alignments=alignments, coverage_score=1.0,
    )


def evaluate_discriminative_critical_path(dag: TaskDAG, alignment: AlignmentMap, budget: int):
    """Evaluate the discriminative critical-path strategy."""
    selected = _select_subtasks(dag, alignment, budget)
    return selected


def compute_info_gain(prior: GaussianPrior, probe: Probe, score: float) -> float:
    """Compute information gain (reduction in entropy) from a probe."""
    posterior = bayesian_update(prior, probe, score)
    # Info gain ~ reduction in variance
    prior_entropy = math.log(prior.sigma)
    posterior_entropy = math.log(posterior.sigma)
    return prior_entropy - posterior_entropy


def main():
    parser = argparse.ArgumentParser(description="Evaluate probe selection strategies")
    parser.add_argument("--budget", type=int, default=2, help="Probe budget per agent")
    args = parser.parse_args()

    dag = _synthetic_dag()
    alignment = _synthetic_alignment("test-server")

    print(f"\n{'='*60}")
    print("PROBE SELECTION STRATEGY EVALUATION")
    print(f"{'='*60}\n")
    print(f"TaskDAG: {len(dag.nodes)} subtasks, difficulty={dag.estimated_difficulty:.2f}")
    print(f"Budget: {args.budget} probes\n")

    # Strategy: discriminative critical-path
    selected = evaluate_discriminative_critical_path(dag, alignment, args.budget)
    print(f"Strategy: discriminative_critical_path")
    print(f"  Selected {len(selected)} subtasks:")
    for node, align in selected:
        print(f"    {node.id}: difficulty={node.difficulty:.2f}, "
              f"discriminative={node.is_discriminative}, "
              f"tool={align.tool_name}, match={align.match_type}")

    # Simulate scoring with different agent abilities
    print(f"\n{'='*60}")
    print("SCORING SIMULATION")
    print(f"{'='*60}\n")

    prior = GaussianPrior(mu=0.5, sigma=0.5)

    for true_ability in [0.2, 0.5, 0.8]:
        print(f"\nTrue agent ability: {true_ability}")
        probes = []
        scores = []

        for node, align in selected:
            probe = Probe(
                probe_id=f"P_{node.id}",
                targets_subtask=node.id,
                tool=align.tool_name,
                arguments={"test": True},
                estimated_difficulty=node.difficulty,
                discrimination=1.2,
                rubric=_make_rubric(),
                timeout_seconds=15,
                priority="PRIMARY",
            )
            # Simulate: agent succeeds if ability > difficulty
            score = 1.0 if true_ability > node.difficulty else 0.0
            probes.append(probe)
            scores.append(score)

            info = compute_info_gain(prior, probe, score)
            print(f"  Probe {probe.probe_id}: score={score:.1f}, info_gain={info:.4f}")

        estimate = score_agent(prior, probes, scores)
        print(f"  Result: theta={estimate.theta:.3f}, sigma={estimate.sigma:.3f}, "
              f"confidence={estimate.confidence:.3f}, tier={estimate.testability_tier}")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
