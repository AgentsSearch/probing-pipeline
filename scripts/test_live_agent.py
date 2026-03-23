"""End-to-end test: pipeline + real probe executor on localhost:8001."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

import httpx

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm.client import LLMClient
from src.models.integration import (
    ActionStep,
    CandidateAgent,
    InlineTool,
    LLMExtracted,
    ProbeExecutionResult,
    RemoteEndpoint,
    RetrievalResult,
)
from src.pipeline import PipelineConfig, ProbePipeline
from src.tool_index.indexer import ToolIndexer, ToolRecord
from src.tool_index.retriever import ToolRetriever

EXECUTOR_URL = os.environ.get("EXECUTOR_URL", "http://localhost:8001")

# ── The lilo agent ──────────────────────────────────────────────────────────

LILO_AGENT = CandidateAgent(
    agent_id="9d9fd884fe3f4048",
    score=0.75,
    remotes=[RemoteEndpoint(type="streamable-http", url="https://mcp.lilo.property/mcp")],
    description="Vacation rental booking and protection for AI agents. Instant API key, 10 free credits.",
    documentation_quality=0.65,
    tools=[
        InlineTool(name="search_properties", description="Search for vacation rental properties by location. Returns bookable properties with verified host protection.", input_schema={"type": "object", "properties": {"location": {"type": "string", "description": "Location to search (city, state, or address)"}, "property_type": {"type": "string"}, "limit": {"type": "number"}, "verified_only": {"type": "boolean"}}, "required": ["location"]}),
        InlineTool(name="check_availability", description="Check real-time availability and pricing for a property on specific dates.", input_schema={"type": "object", "properties": {"property_id": {"type": "string"}, "check_in_date": {"type": "string"}, "check_out_date": {"type": "string"}, "guest_count": {"type": "number"}}, "required": ["property_id", "check_in_date", "check_out_date"]}),
        InlineTool(name="get_property", description="Get detailed information about a lilo-protected property by its ID or lilo_code.", input_schema={"type": "object", "properties": {"property_id": {"type": "string"}}, "required": ["property_id"]}),
        InlineTool(name="natural_language_search", description="Search properties using natural language. Example: 'romantic beachfront getaway with hot tub'.", input_schema={"type": "object", "properties": {"query": {"type": "string"}, "threshold": {"type": "number"}, "limit": {"type": "number"}}, "required": ["query"]}),
        InlineTool(name="get_network_stats", description="Get lilo network statistics - total properties protected, evidence sealed, etc.", input_schema={"type": "object", "properties": {}, "required": []}),
        InlineTool(name="check_protection_status", description="Check if a property is actively protected by lilo.", input_schema={"type": "object", "properties": {"property_id": {"type": "string"}}, "required": ["property_id"]}),
        InlineTool(name="get_jurisdiction_rules", description="Get local STR regulations for a jurisdiction.", input_schema={"type": "object", "properties": {"city": {"type": "string"}, "state": {"type": "string"}, "property_type": {"type": "string"}}, "required": ["state"]}),
        InlineTool(name="get_demand_forecast", description="Forecast booking demand for a location.", input_schema={"type": "object", "properties": {"location": {"type": "string"}, "date_range_start": {"type": "string"}, "date_range_end": {"type": "string"}}, "required": ["location"]}),
    ],
    llm_extracted=LLMExtracted(
        capabilities=[
            "Provide vacation rental property search by location",
            "Check real-time pricing and availability dates for rentals",
            "Create bookings with instant confirmation",
            "Assess guest risk scores before booking",
        ],
        limitations=[],
        requirements=["API key required for access"],
    ),
    is_available=True,
)

QUERY = "Find a vacation rental in Miami for 4 guests, check availability for next weekend, and get the local regulations"


def build_index_from_agent(agent: CandidateAgent) -> str:
    """Build a FAISS index from the agent's inline tools."""
    indexer = ToolIndexer(embedding_model="all-MiniLM-L6-v2")
    for tool in agent.tools:
        indexer.add_tools([
            ToolRecord(
                tool_name=tool.name,
                server_id=agent.agent_id,
                description=tool.description,
                parameter_schema=tool.input_schema,
            )
        ])
    indexer.build_index()
    index_dir = tempfile.mkdtemp(prefix="probe_live_test_")
    indexer.save(index_dir)
    return index_dir


async def execute_probes(plan, agent: CandidateAgent) -> list[ProbeExecutionResult]:
    """Send probes to the real executor."""
    payload = {
        "agent_id": agent.agent_id,
        "probes": [
            {
                "probe_id": p.probe_id,
                "targets_subtask": p.targets_subtask,
                "tool": p.tool,
                "arguments": p.arguments,
                "estimated_difficulty": p.estimated_difficulty,
                "discrimination": p.discrimination,
                "rubric": [
                    {"name": r.name, "weight": r.weight, "criteria": r.criteria, "pass_threshold": r.pass_threshold}
                    for r in p.rubric
                ],
                "timeout_seconds": p.timeout_seconds,
                "priority": p.priority,
            }
            for p in plan.probes
        ],
        "total_timeout": plan.total_budget_seconds,
        "remotes": [{"type": r.type, "url": r.url} for r in agent.remotes],
    }

    print(f"\n{'='*60}")
    print("EXECUTOR REQUEST")
    print(f"{'='*60}")
    print(json.dumps(payload, indent=2))

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(f"{EXECUTOR_URL}/execute", json=payload)
        resp.raise_for_status()
        data = resp.json()

    print(f"\n{'='*60}")
    print("EXECUTOR RESPONSE")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2))

    results = []
    for item in data:
        trajectory = [
            ActionStep(
                action=s.get("action", ""),
                tool_name=s.get("tool_name"),
                arguments=s.get("arguments"),
                result=s.get("result"),
                error=s.get("error"),
                latency_ms=s.get("latency_ms", 0),
            )
            for s in item.get("trajectory", [])
        ]
        results.append(
            ProbeExecutionResult(
                agent_id=item["agent_id"],
                probe_id=item["probe_id"],
                output=item.get("output"),
                trajectory=trajectory,
                latency_ms=item.get("latency_ms", 0),
                success=item.get("success", False),
                error_info=item.get("error_info"),
            )
        )
    return results


async def main():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("ERROR: CEREBRAS_API_KEY not set")
        sys.exit(1)

    # Check executor health
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{EXECUTOR_URL}/health")
            print(f"Executor health: {r.json()}")
        except Exception as e:
            print(f"ERROR: Cannot reach executor at {EXECUTOR_URL}: {e}")
            sys.exit(1)

    # Build FAISS index from agent's tools
    print("\nBuilding FAISS index from agent tools...")
    index_dir = build_index_from_agent(LILO_AGENT)

    # Init pipeline
    llm = LLMClient.from_config(api_key=api_key)
    retriever = ToolRetriever(index_dir)
    pipeline = ProbePipeline(
        llm=llm,
        retriever=retriever,
        config=PipelineConfig(probe_budget=2, total_timeout=60),
    )

    # Stage 1: Task Analysis
    print(f"\n{'='*60}")
    print(f"STAGE 1: Task Analysis")
    print(f"Query: {QUERY}")
    print(f"{'='*60}")
    dag, t1 = pipeline.run_stage_1(QUERY)
    if dag is None:
        print("Stage 1 FAILED")
        sys.exit(1)
    print(f"  Intent: {dag.intent}")
    print(f"  Domain: {dag.domain}")
    print(f"  Difficulty: {dag.estimated_difficulty}")
    print(f"  Nodes ({len(dag.nodes)}):")
    for n in dag.nodes:
        print(f"    {n.id}: {n.description} (d={n.difficulty}, disc={n.is_discriminative})")
    print(f"  Critical path: {dag.critical_path}")
    print(f"  Time: {t1:.2f}s")

    # Stages 2-4
    print(f"\n{'='*60}")
    print(f"STAGES 2-4: Alignment → Generation → Validation")
    print(f"{'='*60}")
    result = pipeline.run_stages_2_to_4_for_agent(dag, LILO_AGENT)

    if result.error_code:
        print(f"  ERROR at {result.error_code}: {result.error_detail}")
        sys.exit(1)

    print(f"\n  Stage 2 (Alignment):")
    print(f"    Coverage: {result.alignment.coverage_score:.2f}")
    print(f"    Tools evaluated: {result.alignment.tools_evaluated}")
    print(f"    Unmatched subtasks: {result.alignment.unmatched_subtasks}")
    for a in result.alignment.alignments:
        print(f"    {a.subtask_id} → {a.tool_name} ({a.match_type}, conf={a.confidence:.2f})")
    print(f"    Time: {result.timings.get('stage_2_alignment', 0):.2f}s")

    print(f"\n  Stage 3 (Probe Generation):")
    print(f"    Probes generated: {len(result.probe_plan.probes)}")
    for p in result.probe_plan.probes:
        print(f"    {p.probe_id}: tool={p.tool}, subtask={p.targets_subtask}, d={p.estimated_difficulty}, a={p.discrimination}")
        print(f"      args={json.dumps(p.arguments)}")
    print(f"    Time: {result.timings.get('stage_3_generation', 0):.2f}s")

    print(f"\n  Stage 4 (Validation):")
    print(f"    Probes validated: {len(result.validated_plan.probes)}")
    print(f"    Time: {result.timings.get('stage_4_validation', 0):.2f}s")

    if not result.validated_plan.probes:
        print("\n  No valid probes — skipping execution and scoring")
        sys.exit(0)

    # Execution via probe-runner
    print(f"\n{'='*60}")
    print(f"STAGE 5: Probe Execution (via {EXECUTOR_URL})")
    print(f"{'='*60}")
    exec_results = await execute_probes(result.validated_plan, LILO_AGENT)
    print(f"\n  Results: {len(exec_results)}")
    for er in exec_results:
        status = "PASS" if er.success else "FAIL"
        print(f"    {er.probe_id}: {status} ({er.latency_ms}ms)")
        if er.error_info:
            print(f"      Error: {er.error_info}")

    # Scoring
    print(f"\n{'='*60}")
    print(f"STAGE 6: Bayesian IRT Scoring")
    print(f"{'='*60}")
    ranked = pipeline.score_agent_results(result, exec_results)
    print(f"  Agent: {ranked.agent_id}")
    print(f"  Theta: {ranked.theta:.4f}")
    print(f"  Sigma: {ranked.sigma:.4f}")
    print(f"  Confidence: {ranked.confidence:.4f}")
    print(f"  Testability: {ranked.testability_tier}")
    print(f"  Prior influence: {ranked.prior_influence:.4f}")
    print(f"  Summary: {ranked.probe_summary}")
    print(f"\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
