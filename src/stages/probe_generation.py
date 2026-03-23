"""Stage 3: Probe Plan Generation — produce executable probes from alignments.

Input:  TaskDAG + AlignmentMap
Output: ProbePlan with 1-3 executable Probes
Cost:   0-1 LLM calls (0 if template cache hit)

Strategy: Discriminative Critical-Path (tiered)
  Tier 1: is_discriminative AND match_type in {direct, partial} — strict
  Tier 2: is_discriminative AND match_type == "inferred" — inferred promotion
  Tier 3: ANY subtask with match_type in {direct, partial, inferred} — relaxed fallback
  Within each tier: sort by difficulty descending, select top-N (N = budget)
  Sanity check: if all selected are high-difficulty and budget allows, add a low-difficulty probe
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.llm.client import LLMClient
from src.models.alignment import AlignmentMap, ToolAlignment
from src.models.probe import Probe, ProbePlan, RubricDimension
from src.models.task import SubtaskNode, TaskDAG
from src.templates.library import TemplateLibrary

logger = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "config"
    / "prompts"
    / "probe_generation.txt"
)

_HIGH_DIFFICULTY_THRESHOLD = 0.6


def _load_prompt_template() -> str:
    with open(_PROMPT_PATH) as f:
        return f.read()


def _select_subtasks(
    dag: TaskDAG,
    alignment: AlignmentMap,
    budget: int,
    limitations: list[str] | None = None,
) -> list[tuple[SubtaskNode, ToolAlignment]]:
    """Select which subtasks to probe using the discriminative critical-path strategy.

    Args:
        dag: Task DAG from Stage 1.
        alignment: Alignment map from Stage 2.
        budget: Maximum number of probes.

    Returns:
        List of (subtask, best_alignment) pairs to probe.
    """
    candidates: list[tuple[SubtaskNode, ToolAlignment]] = []

    # Tier 1 (strict): discriminative subtasks with direct/partial matches
    for node in dag.nodes:
        if not node.is_discriminative:
            continue
        best = alignment.best_alignment_for_subtask(node.id)
        if best is None or best.match_type not in ("direct", "partial"):
            continue
        candidates.append((node, best))

    # Demote subtasks that overlap with known limitations (sort lower, don't exclude)
    limitation_keywords: set[str] = set()
    if limitations:
        for lim in limitations:
            limitation_keywords.update(lim.lower().split())

    def _sort_key(pair: tuple[SubtaskNode, ToolAlignment]) -> tuple[bool, float]:
        node = pair[0]
        overlaps = False
        if limitation_keywords:
            desc_words = set(node.description.lower().split())
            overlaps = bool(desc_words & limitation_keywords)
        # Non-overlapping first (False < True), then by difficulty descending
        return (overlaps, -node.difficulty)

    candidates.sort(key=_sort_key)

    selected = candidates[:budget]

    # If all selected are high-difficulty and budget allows, add a sanity check
    if (
        len(selected) < budget
        and selected
        and all(n.difficulty >= _HIGH_DIFFICULTY_THRESHOLD for n, _ in selected)
    ):
        # Find a low-difficulty non-discriminative subtask with a tool match
        for node in dag.nodes:
            if node.difficulty < _HIGH_DIFFICULTY_THRESHOLD:
                best = alignment.best_alignment_for_subtask(node.id)
                if best and best.match_type in ("direct", "partial"):
                    selected.append((node, best))
                    break

    # Tier 2: if Tier 1 found nothing, try discriminative subtasks with "inferred" matches
    if not selected:
        for node in dag.nodes:
            if not node.is_discriminative:
                continue
            best = alignment.best_alignment_for_subtask(node.id)
            if best is not None and best.match_type == "inferred":
                selected.append((node, best))
                if len(selected) >= budget:
                    break
        selected.sort(key=_sort_key)
        selected = selected[:budget]

    # Tier 3 (relaxed fallback): any subtask with direct/partial/inferred match
    if not selected:
        for node in dag.nodes:
            best = alignment.best_alignment_for_subtask(node.id)
            if best and best.match_type in ("direct", "partial", "inferred"):
                selected.append((node, best))
                if len(selected) >= budget:
                    break

    return selected


def _probe_from_template(
    template_match: Any,
    node: SubtaskNode,
    alignment: ToolAlignment,
    query: str,
    probe_index: int,
) -> Probe:
    """Instantiate a probe from a template match."""
    return Probe(
        probe_id=f"P{probe_index}",
        targets_subtask=node.id,
        tool=alignment.tool_name,
        arguments=template_match.arg_template,
        estimated_difficulty=template_match.difficulty,
        discrimination=template_match.discrimination,
        rubric=list(template_match.rubric_template),
        timeout_seconds=30 if node.difficulty > 0.5 else 15,
        priority="PRIMARY" if node.is_discriminative else "SECONDARY",
    )


def _probe_from_llm(
    node: SubtaskNode,
    alignment: ToolAlignment,
    query: str,
    llm: LLMClient,
    probe_index: int,
    limitations: list[str] | None = None,
) -> Probe:
    """Generate a probe via LLM when no template is available."""
    template = _load_prompt_template()

    known_limitations = ""
    if limitations:
        known_limitations = "\n".join(f"- {lim}" for lim in limitations)

    prompt = (
        template
        .replace("{subtask_description}", node.description)
        .replace("{difficulty}", str(node.difficulty))
        .replace("{tool_name}", alignment.tool_name)
        .replace("{tool_description}", alignment.tool_description or f"Tool on server {alignment.server_id}")
        .replace("{parameter_schema}", json.dumps(alignment.tool_parameter_schema, indent=2) if alignment.tool_parameter_schema else json.dumps(dict(alignment.parameter_mapping), default=str))
        .replace("{query}", query)
        .replace("{known_limitations}", known_limitations)
    )

    data = llm.complete_json(
        prompt,
        system="You are a probe generation engine. Return only valid JSON.",
    )

    rubric = [
        RubricDimension(
            name=r["name"],
            weight=float(r["weight"]),
            criteria=r["criteria"],
            pass_threshold=r["pass_threshold"],
        )
        for r in data.get("rubric", [])
    ]

    # Ensure at least 2 rubric dimensions
    if len(rubric) < 2:
        rubric = [
            RubricDimension(name="correctness", weight=0.6, criteria="Output is factually correct", pass_threshold="No factual errors"),
            RubricDimension(name="completeness", weight=0.4, criteria="Output addresses the full request", pass_threshold="All requested information present"),
        ]

    return Probe(
        probe_id=f"P{probe_index}",
        targets_subtask=node.id,
        tool=alignment.tool_name,
        arguments=data.get("arguments", {}),
        estimated_difficulty=float(data.get("estimated_difficulty", node.difficulty)),
        discrimination=float(data.get("discrimination", 1.0)),
        rubric=rubric,
        timeout_seconds=int(data.get("timeout_seconds", 30 if node.difficulty > 0.5 else 15)),
        priority=data.get("priority", "PRIMARY"),
    )


def generate_probe_plan(
    query: str,
    dag: TaskDAG,
    alignment: AlignmentMap,
    llm: LLMClient,
    template_library: TemplateLibrary | None = None,
    budget: int = 2,
    total_timeout: int = 30,
    limitations: list[str] | None = None,
) -> ProbePlan:
    """Generate a probe plan for a single agent.

    Args:
        query: Original user query.
        dag: TaskDAG from Stage 1.
        alignment: AlignmentMap from Stage 2 for this agent.
        llm: Initialised LLMClient.
        template_library: Optional template library for cache hits.
        budget: Maximum number of probes.
        total_timeout: Total wall-clock budget in seconds.
        limitations: Optional list of known agent limitations from LLM extraction.

    Returns:
        ProbePlan with 1-budget executable probes.
    """
    selected = _select_subtasks(dag, alignment, budget, limitations=limitations)

    if not selected:
        n_align = len(alignment.alignments)
        by_type: dict[str, int] = {}
        for a in alignment.alignments:
            by_type[a.match_type] = by_type.get(a.match_type, 0) + 1
        n_disc = sum(1 for n in dag.nodes if n.is_discriminative)
        logger.warning(
            "No probeable subtasks for agent %s: %d alignments %s, %d/%d discriminative",
            alignment.agent_id, n_align, by_type, n_disc, len(dag.nodes),
        )
        return ProbePlan(
            query=query,
            agent_id=alignment.agent_id,
            strategy="discriminative_critical_path",
            probes=[],
            total_budget_seconds=total_timeout,
        )

    probes: list[Probe] = []
    seen_probes: set[str] = set()  # (tool_name, args_json) for dedup
    for i, (node, tool_align) in enumerate(selected, start=1):
        # Try template library first
        if template_library:
            tmpl = template_library.lookup(tool_align.tool_name, node.difficulty)
            if tmpl:
                logger.info(
                    "Template hit for %s (difficulty=%.2f)", tool_align.tool_name, node.difficulty
                )
                probe = _probe_from_template(tmpl, node, tool_align, query, i)
                dedup_key = f"{probe.tool}:{json.dumps(probe.arguments, sort_keys=True)}"
                if dedup_key in seen_probes:
                    logger.info("Skipping duplicate probe for %s", probe.tool)
                    continue
                seen_probes.add(dedup_key)
                probes.append(probe)
                continue

        # Fall back to LLM generation
        try:
            probe = _probe_from_llm(node, tool_align, query, llm, i, limitations=limitations)
            dedup_key = f"{probe.tool}:{json.dumps(probe.arguments, sort_keys=True)}"
            if dedup_key in seen_probes:
                logger.info("Skipping duplicate probe for %s", probe.tool)
                continue
            seen_probes.add(dedup_key)
            probes.append(probe)
        except Exception as e:
            logger.error("Failed to generate probe for subtask %s: %s", node.id, e)

    plan = ProbePlan(
        query=query,
        agent_id=alignment.agent_id,
        strategy="discriminative_critical_path",
        probes=probes,
        total_budget_seconds=total_timeout,
    )

    logger.info(
        "Generated probe plan for %s: %d probes",
        alignment.agent_id,
        len(probes),
    )
    return plan
