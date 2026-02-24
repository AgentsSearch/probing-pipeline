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

from src.llm.client import LLMClient
from src.models.alignment import AlignmentMap, ParameterMap, ToolAlignment
from src.models.task import TaskDAG
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
    subtask_description: str,
    capability: str,
    candidates: list[ToolCandidate],
) -> str:
    """Fill in the reranking prompt template with subtask and tool details."""
    tools_data = [
        {
            "tool_name": c.tool_name,
            "server_id": c.server_id,
            "description": c.description,
            "parameter_schema": c.parameter_schema,
            "capability_tags": c.capability_tags,
        }
        for c in candidates
    ]

    return (
        template
        .replace("{subtask_description}", subtask_description)
        .replace("{capability}", capability)
        .replace("{tools_json}", json.dumps(tools_data, indent=2))
    )


def _parse_rerank_response(
    data: dict[str, Any],
    retrieval_scores: dict[str, float],
) -> list[ToolAlignment]:
    """Parse LLM reranking response into ToolAlignment objects."""
    alignments: list[ToolAlignment] = []

    for item in data.get("alignments", []):
        tool_name = item["tool_name"]
        param_mapping = {}
        for key, val in item.get("parameter_mapping", {}).items():
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
) -> AlignmentMap:
    """Run tool-task alignment for a single candidate agent.

    Phase A: For each subtask, retrieve top-k tools from the FAISS index
             filtered to the agent's server.
    Phase B: LLM reranks the shortlisted tools for precise alignment.

    Args:
        dag: The decomposed TaskDAG from Stage 1.
        agent_server_id: Server ID of the candidate agent.
        retriever: Initialised ToolRetriever with loaded FAISS index.
        llm: Initialised LLMClient.
        retrieval_k: Number of tools to retrieve per subtask from FAISS.

    Returns:
        AlignmentMap with alignments and coverage score for this agent.
    """
    template = _load_prompt_template()
    all_alignments: list[ToolAlignment] = []
    unmatched: list[str] = []

    candidate_servers = {agent_server_id}

    for node in dag.nodes:
        # Phase A: embedding retrieval
        candidates = retriever.retrieve(
            query=node.description,
            candidate_server_ids=candidate_servers,
            k=retrieval_k,
        )

        if not candidates:
            logger.info(
                "No tools found for subtask %s on server %s",
                node.id,
                agent_server_id,
            )
            unmatched.append(node.id)
            continue

        # Shortlist for reranker
        shortlist = candidates[:_RERANK_SHORTLIST]
        retrieval_scores = {c.tool_name: c.similarity_score for c in shortlist}

        # Phase B: LLM reranking
        prompt = _build_rerank_prompt(
            template, node.description, node.capability, shortlist
        )

        try:
            data = llm.complete_json(
                prompt,
                system="You are a tool-task alignment engine. Return only valid JSON.",
            )
            alignments = _parse_rerank_response(data, retrieval_scores)

            # Filter out "none" matches
            valid = [a for a in alignments if a.match_type != "none"]
            if valid:
                all_alignments.extend(valid)
            else:
                unmatched.append(node.id)

        except Exception as e:
            logger.error(
                "Reranking failed for subtask %s: %s", node.id, e
            )
            unmatched.append(node.id)

    # Compute coverage
    matched_subtasks = {a.subtask_id for a in all_alignments}
    total_subtasks = len(dag.nodes)
    coverage = len(matched_subtasks) / total_subtasks if total_subtasks > 0 else 0.0

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
