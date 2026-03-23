"""Stage 2: Tool-Task Alignment — match TaskDAG subtasks to MCP tools.

Input:  TaskDAG + candidate agent IDs
Output: AlignmentMap per agent
Cost:   FAISS queries (cheap) + 1 LLM call per agent (reranker)

Two phases:
  A) Embedding retrieval via FAISS tool index
  B) LLM reranking for precise alignment assessment
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss

from src.llm.client import LLMClient
from src.models.alignment import AlignmentMap, ParameterMap, ToolAlignment
from src.models.task import TaskDAG
from src.tool_index.indexer import ToolRecord
from src.tool_index.retriever import ToolCandidate, ToolRetriever

logger = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "config"
    / "prompts"
    / "tool_alignment.txt"
)

_RERANK_SHORTLIST = 8  # Max tools per subtask sent to LLM reranker


def _load_prompt_template() -> str:
    with open(_PROMPT_PATH) as f:
        return f.read()


def _build_rerank_prompt(
    template: str,
    subtasks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    agent_description: str | None = None,
    agent_capabilities: list[str] | None = None,
) -> str:
    """Fill in the reranking prompt template with all subtasks and deduplicated tools.

    Args:
        template: The prompt template with {subtasks_json}, {tools_json},
            and optional {agent_context} placeholders.
        subtasks: List of subtask dicts with id, description, capability.
        candidates: Deduplicated list of tool dicts for the prompt.
        agent_description: Optional agent description for context.
        agent_capabilities: Optional list of agent capabilities.

    Returns:
        Filled prompt string.
    """
    agent_context = ""
    if agent_description or agent_capabilities:
        parts = []
        if agent_description:
            parts.append(f"Agent description: {agent_description}")
        if agent_capabilities:
            parts.append(f"Agent capabilities: {', '.join(agent_capabilities)}")
        agent_context = "\n".join(parts)

    return (
        template
        .replace("{subtasks_json}", json.dumps(subtasks, indent=2))
        .replace("{tools_json}", json.dumps(candidates, indent=2))
        .replace("{agent_context}", agent_context)
    )


def _parse_rerank_response(
    data: dict[str, Any],
    retrieval_scores: dict[str, float],
) -> list[ToolAlignment]:
    """Parse LLM reranking response into ToolAlignment objects."""
    alignments: list[ToolAlignment] = []

    for item in data.get("alignments", []):
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name")
        if not tool_name or not item.get("subtask_id") or not item.get("server_id"):
            logger.warning("Skipping alignment entry with missing required fields: %s", item)
            continue

        param_mapping = {}
        raw_mapping = item.get("parameter_mapping", {})
        if isinstance(raw_mapping, dict):
            for key, val in raw_mapping.items():
                if isinstance(val, dict):
                    param_mapping[key] = ParameterMap(
                        subtask_param=val.get("subtask_param", key),
                        tool_param=val.get("tool_param", key),
                        transform=val.get("transform"),
                    )

        alignments.append(
            ToolAlignment(
                subtask_id=item["subtask_id"],
                tool_name=tool_name,
                server_id=item["server_id"],
                match_type=item.get("match_type", "none"),
                confidence=float(item.get("confidence", 0.0)),
                retrieval_score=retrieval_scores.get(tool_name, 0.0),
                rerank_score=float(item.get("rerank_score", item.get("confidence", 0.0))),
                parameter_mapping=param_mapping,
            )
        )

    return alignments


def align_tools_for_agent(
    dag: TaskDAG,
    agent_server_id: str,
    retriever: ToolRetriever,
    llm: LLMClient,
    retrieval_k: int = 20,
    agent_description: str | None = None,
    agent_capabilities: list[str] | None = None,
    extra_index: tuple[faiss.Index, list[ToolRecord]] | None = None,
) -> AlignmentMap:
    """Run tool-task alignment for a single candidate agent.

    Phase A: For each subtask, retrieve top-k tools from the FAISS index
             filtered to the agent's server. Collect and deduplicate.
    Phase B: Single batched LLM call reranks all tools against all subtasks.

    Args:
        dag: The decomposed TaskDAG from Stage 1.
        agent_server_id: Server ID of the candidate agent.
        retriever: Initialised ToolRetriever with loaded FAISS index.
        llm: Initialised LLMClient.
        retrieval_k: Number of tools to retrieve per subtask from FAISS.
        agent_description: Optional agent description for richer context.
        agent_capabilities: Optional list of agent capabilities.

    Returns:
        AlignmentMap with alignments and coverage score for this agent.
    """
    template = _load_prompt_template()
    candidate_servers = {agent_server_id}

    # --- Phase A: FAISS retrieval for all subtasks, then deduplicate ---
    # Track which subtasks had no FAISS results at all
    nodes_with_candidates: list[str] = []
    unmatched: list[str] = []

    # Union of all tools across subtasks (keep highest similarity per tool)
    merged_tools: dict[str, ToolCandidate] = {}
    retrieval_scores: dict[str, float] = {}

    for node in dag.nodes:
        candidates = retriever.retrieve(
            query=node.description,
            candidate_server_ids=candidate_servers,
            k=retrieval_k,
            extra_index=extra_index,
        )

        if not candidates:
            logger.info(
                "No tools found for subtask %s on server %s",
                node.id,
                agent_server_id,
            )
            unmatched.append(node.id)
            continue

        nodes_with_candidates.append(node.id)

        for c in candidates[:_RERANK_SHORTLIST]:
            existing = merged_tools.get(c.tool_name)
            if existing is None or c.similarity_score > existing.similarity_score:
                merged_tools[c.tool_name] = c
            # Keep highest retrieval score per tool
            retrieval_scores[c.tool_name] = max(
                retrieval_scores.get(c.tool_name, 0.0),
                c.similarity_score,
            )

    # If no subtask had any FAISS results, return empty alignment
    if not nodes_with_candidates:
        return AlignmentMap(
            agent_id=agent_server_id,
            server_tool_count=len(retriever.tools),
            tools_evaluated=0,
            alignments=[],
            coverage_score=0.0,
            unmatched_subtasks=[n.id for n in dag.nodes],
        )

    # Build tool metadata lookup (description, schema) from merged candidates
    tool_meta = {
        c.tool_name: (c.description, c.parameter_schema)
        for c in merged_tools.values()
    }

    # Build prompt data
    subtasks_data = [
        {"id": node.id, "description": node.description, "capability": node.capability}
        for node in dag.nodes
        if node.id in nodes_with_candidates
    ]
    tools_data = [
        {
            "tool_name": c.tool_name,
            "server_id": c.server_id,
            "description": c.description,
            "parameter_schema": c.parameter_schema,
            "capability_tags": c.capability_tags,
        }
        for c in merged_tools.values()
    ]

    # --- Phase B: single batched LLM reranking call ---
    prompt = _build_rerank_prompt(
        template, subtasks_data, tools_data,
        agent_description=agent_description,
        agent_capabilities=agent_capabilities,
    )

    valid_node_ids = {node.id for node in dag.nodes}
    all_alignments: list[ToolAlignment] = []

    try:
        data = llm.complete_json(
            prompt,
            system="You are a tool-task alignment engine. Return only valid JSON.",
            max_tokens=8192,
        )
        alignments = _parse_rerank_response(data, retrieval_scores)

        # Validate subtask_ids and populate tool metadata
        for a in alignments:
            # Ensure subtask_id is a known node ID
            if a.subtask_id not in valid_node_ids:
                logger.warning(
                    "LLM returned unknown subtask_id '%s', skipping alignment",
                    a.subtask_id,
                )
                continue
            if a.tool_name in tool_meta:
                a.tool_description, a.tool_parameter_schema = tool_meta[a.tool_name]
            if a.match_type != "none":
                all_alignments.append(a)

    except Exception as e:
        logger.error("Batched reranking failed for %s: %s", agent_server_id, e)
        # All subtasks that had candidates are now unmatched
        unmatched.extend(nodes_with_candidates)

    # Check for subtasks with candidates but no valid alignments
    matched_subtasks = {a.subtask_id for a in all_alignments}
    for node_id in nodes_with_candidates:
        if node_id not in matched_subtasks and node_id not in unmatched:
            unmatched.append(node_id)

    # Compute coverage weighted by best match confidence per subtask
    # A subtask matched with confidence 0.25 contributes 0.25, not 1.0
    total_subtasks = len(dag.nodes)
    if total_subtasks > 0 and all_alignments:
        best_conf: dict[str, float] = {}
        for a in all_alignments:
            if a.subtask_id not in best_conf or a.confidence > best_conf[a.subtask_id]:
                best_conf[a.subtask_id] = a.confidence
        coverage = sum(best_conf.values()) / total_subtasks
    else:
        coverage = 0.0
    tools_evaluated = len({a.tool_name for a in all_alignments})

    alignment_map = AlignmentMap(
        agent_id=agent_server_id,
        server_tool_count=len(retriever.tools),
        tools_evaluated=tools_evaluated,
        alignments=all_alignments,
        coverage_score=coverage,
        unmatched_subtasks=unmatched,
    )

    logger.info(
        "Alignment for %s: coverage=%.2f, matched=%d/%d subtasks, %d tools",
        agent_server_id,
        coverage,
        len(matched_subtasks),
        total_subtasks,
        tools_evaluated,
    )

    return alignment_map
