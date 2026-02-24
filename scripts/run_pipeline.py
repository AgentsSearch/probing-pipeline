#!/usr/bin/env python3
"""Run the probing pipeline end-to-end against sample MCP servers.

Usage:
    export CEREBRAS_API_KEY=your_key_here
    python scripts/run_pipeline.py --query "Find the current weather in London"

Or with a different provider:
    export OPENAI_API_KEY=your_key_here
    python scripts/run_pipeline.py \
        --query "Find the current weather in London" \
        --base-url https://api.openai.com/v1 \
        --model gpt-4o-mini \
        --api-key-env OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import LLMClient
from src.models.integration import (
    ActionStep,
    CandidateAgent,
    ProbeExecutionResult,
    RetrievalResult,
)
from src.pipeline import ProbePipeline, PipelineConfig
from src.tool_index.indexer import ToolIndexer, ToolRecord
from src.tool_index.retriever import ToolRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def build_index(servers_path: str | Path) -> str:
    """Build a FAISS index from MCP server definitions and return the temp dir."""
    with open(servers_path) as f:
        servers = json.load(f)

    indexer = ToolIndexer(embedding_model="all-MiniLM-L6-v2")

    for server in servers:
        for tool in server["tools"]:
            indexer.add_tools([
                ToolRecord(
                    tool_name=tool["tool_name"],
                    server_id=server["server_id"],
                    description=tool["description"],
                    capability_tags=tool.get("capability_tags", []),
                    parameter_schema=tool.get("parameter_schema", {}),
                    output_schema=tool.get("output_schema", {}),
                    complexity_estimate=tool.get("complexity_estimate", 0.5),
                )
            ])

    indexer.build_index()

    index_dir = tempfile.mkdtemp(prefix="probe_index_")
    indexer.save(index_dir)
    logger.info("Built FAISS index at %s (%d tools)", index_dir, len(indexer.tools))
    return index_dir


def simulate_execution(plan, server_url: str) -> list[ProbeExecutionResult]:
    """Simulate probe execution (placeholder for Stream C integration).

    In production, this would send probes to the MCP sandbox.
    For now, simulates successful execution with dummy output.
    """
    results = []
    for probe in plan.probes:
        results.append(
            ProbeExecutionResult(
                agent_id=plan.agent_id,
                probe_id=probe.probe_id,
                output={"status": "simulated", "message": f"Simulated result for {probe.tool}"},
                trajectory=[
                    ActionStep(action="call_tool", tool_name=probe.tool, result="simulated_ok"),
                ],
                latency_ms=150,
                success=True,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the probing pipeline")
    parser.add_argument("--query", required=True, help="User query to evaluate")
    parser.add_argument(
        "--servers", default=str(FIXTURES / "sample_mcp_servers.json"),
        help="Path to MCP server definitions JSON",
    )
    parser.add_argument(
        "--api-key-env", default="CEREBRAS_API_KEY",
        help="Environment variable name for the API key (default: CEREBRAS_API_KEY)",
    )
    parser.add_argument("--base-url", default=None, help="Override LLM base URL")
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--budget", type=int, default=2, help="Probe budget per agent")
    parser.add_argument(
        "--simulate-execution", action="store_true", default=True,
        help="Simulate probe execution (default: true, since Stream C not available)",
    )
    args = parser.parse_args()

    # --- API Key ---
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"\nError: Set the {args.api_key_env} environment variable.")
        print(f"  export {args.api_key_env}=your_key_here\n")
        sys.exit(1)

    # --- LLM Client ---
    llm = LLMClient.from_config(api_key=api_key)
    if args.base_url:
        llm.base_url = args.base_url
        llm.__post_init__()  # Re-create the OpenAI client
    if args.model:
        llm.model = args.model

    # --- Build FAISS Index ---
    print(f"\n{'='*60}")
    print("PROBING PIPELINE")
    print(f"{'='*60}")
    print(f"\nQuery: {args.query}")
    print(f"Model: {llm.model}")
    print(f"Budget: {args.budget} probes/agent\n")

    index_dir = build_index(args.servers)
    retriever = ToolRetriever(index_dir)

    # --- Build Candidate Agents from servers ---
    with open(args.servers) as f:
        servers = json.load(f)

    candidates = [
        CandidateAgent(
            agent_id=s["server_id"],
            retrieval_score=0.7,  # placeholder
            mcp_server_url=s["mcp_server_url"],
        )
        for s in servers
    ]

    retrieval_result = RetrievalResult(query=args.query, candidates=candidates)

    # --- Run Pipeline Stages 1-4 ---
    pipeline = ProbePipeline(
        llm=llm,
        retriever=retriever,
        config=PipelineConfig(probe_budget=args.budget),
    )

    print(f"{'─'*60}")
    print("STAGE 1: Task Analysis")
    print(f"{'─'*60}")
    dag, agent_results = pipeline.run_stages_1_to_4(retrieval_result)

    if dag:
        print(f"  Intent: {dag.intent}")
        print(f"  Domain: {dag.domain}")
        print(f"  Subtasks: {len(dag.nodes)}")
        for node in dag.nodes:
            disc = " [DISCRIMINATIVE]" if node.is_discriminative else ""
            print(f"    {node.id}: {node.description} (d={node.difficulty:.2f}){disc}")
        print(f"  Critical path: {dag.critical_path}")
        print(f"  Difficulty: {dag.estimated_difficulty:.2f}")
    else:
        print("  FAILED — see logs")
        sys.exit(1)

    # --- Per-Agent Results ---
    rankings = []
    for result in agent_results:
        agent = result.agent
        print(f"\n{'─'*60}")
        print(f"AGENT: {agent.agent_id}")
        print(f"{'─'*60}")

        if result.error_code:
            print(f"  Error: {result.error_code} — {result.error_detail}")
            ranked = pipeline.score_agent_results(result, [])
            rankings.append(ranked)
            continue

        if result.alignment:
            print(f"  Coverage: {result.alignment.coverage_score:.2f}")
            print(f"  Tools evaluated: {result.alignment.tools_evaluated}")
            print(f"  Unmatched subtasks: {result.alignment.unmatched_subtasks}")

        if result.validated_plan:
            print(f"  Probes: {len(result.validated_plan.probes)}")
            for probe in result.validated_plan.probes:
                print(f"    {probe.probe_id}: {probe.tool} (d={probe.estimated_difficulty:.2f}, "
                      f"a={probe.discrimination:.2f}) [{probe.priority}]")

        # --- Execution ---
        if args.simulate_execution and result.validated_plan:
            exec_results = simulate_execution(result.validated_plan, agent.mcp_server_url)
            print(f"  Execution: {len(exec_results)} probes simulated")
        else:
            exec_results = []

        ranked = pipeline.score_agent_results(result, exec_results)
        rankings.append(ranked)

    # --- Final Rankings ---
    rankings.sort(key=lambda r: r.theta, reverse=True)

    print(f"\n{'='*60}")
    print("FINAL RANKINGS")
    print(f"{'='*60}\n")
    print(f"{'Rank':<5} {'Agent':<25} {'Score':<8} {'±σ':<8} {'Conf':<8} {'Tier':<18} {'Summary'}")
    print(f"{'─'*100}")

    for i, r in enumerate(rankings, 1):
        print(f"{i:<5} {r.agent_id:<25} {r.theta:<8.3f} {r.sigma:<8.3f} "
              f"{r.confidence:<8.3f} {r.testability_tier:<18} {r.probe_summary}")

    # --- Token Usage ---
    tokens = llm.total_tokens()
    print(f"\nToken usage: {tokens['input']} input + {tokens['output']} output = {tokens['total']} total")
    print()


if __name__ == "__main__":
    main()
