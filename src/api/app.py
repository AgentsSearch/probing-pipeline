"""FastAPI application exposing the probing pipeline as an HTTP API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    AgentDetailOut,
    ErrorOut,
    ProbeRequest,
    ProbeResponse,
    RankedAgentOut,
    SubtaskOut,
    TaskDAGOut,
    TokenUsage,
    to_candidate_agent,
)
from src.llm.client import LLMClient
from src.models.integration import (
    ActionStep,
    ProbeExecutionResult,
    RetrievalResult,
)
from src.pipeline import PipelineConfig, ProbePipeline
from src.tool_index.indexer import ToolIndexer, ToolRecord
from src.tool_index.retriever import ToolRetriever

logger = logging.getLogger(__name__)

_FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"

# Module-level state set during lifespan
_pipeline: ProbePipeline | None = None
_llm: LLMClient | None = None
_ready: bool = False


def _build_index(servers_path: str | Path) -> tuple[str, int]:
    """Build a FAISS index from MCP server definitions. Returns (index_dir, tool_count)."""
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
    return index_dir, len(indexer.tools)


_EXECUTOR_URL = os.environ.get("EXECUTOR_URL", "http://localhost:8001")


async def _execute_probes(
    plan, agent_id: str, remotes: list[dict],
) -> list[ProbeExecutionResult]:
    """Send probes to the probe executor at EXECUTOR_URL/execute."""
    payload = {
        "agent_id": agent_id,
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
        "remotes": remotes,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        resp = await client.post(f"{_EXECUTOR_URL}/execute", json=payload)
        resp.raise_for_status()
        data = resp.json()

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: build FAISS index, init LLM client and pipeline."""
    global _pipeline, _llm, _ready

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        logger.error("CEREBRAS_API_KEY not set — pipeline will not be available")
        yield
        return

    _llm = LLMClient.from_config(api_key=api_key)

    servers_json = os.environ.get("SERVERS_JSON", str(_FIXTURES / "sample_mcp_servers.json"))
    index_dir, tool_count = _build_index(servers_json)
    logger.info("FAISS index built: %d tools from %s", tool_count, servers_json)

    retriever = ToolRetriever(index_dir)

    probe_budget = int(os.environ.get("PROBE_BUDGET", "2"))
    total_timeout = int(os.environ.get("TOTAL_TIMEOUT", "30"))

    _pipeline = ProbePipeline(
        llm=_llm,
        retriever=retriever,
        config=PipelineConfig(probe_budget=probe_budget, total_timeout=total_timeout),
    )
    _ready = True
    logger.info("Pipeline ready (budget=%d, timeout=%ds)", probe_budget, total_timeout)

    yield

    _ready = False
    _pipeline = None
    _llm = None


app = FastAPI(
    title="Probing Pipeline API",
    description="Exposes the probing pipeline (Stages 1-4 + scoring) as an HTTP API.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "ready": _ready}


@app.post(
    "/probe",
    response_model=ProbeResponse,
    responses={503: {"model": ErrorOut}, 500: {"model": ErrorOut}},
)
async def probe(request: ProbeRequest):
    """Run the full probing pipeline for the given query and candidates."""
    if not _ready or _pipeline is None or _llm is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready — check CEREBRAS_API_KEY is set",
        )

    t_start = time.monotonic()

    # Convert Pydantic models to dataclasses
    candidates = [to_candidate_agent(c) for c in request.candidates]

    for c in candidates:
        n_tools = len(c.tools) if c.tools else 0
        logger.info("Agent %s: %d tools, available=%s", c.agent_id, n_tools, c.is_available)
    retrieval_result = RetrievalResult(query=request.query, candidates=candidates)

    # Snapshot call_log length for per-request metrics
    log_start = len(_llm.call_log)

    # Run stages 1-4 in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    dag, agent_results = await loop.run_in_executor(
        None, _pipeline.run_stages_1_to_4, retrieval_result
    )

    # Build task_dag response
    task_dag_out = None
    if dag:
        task_dag_out = TaskDAGOut(
            query=dag.query,
            intent=dag.intent,
            domain=dag.domain,
            nodes=[
                SubtaskOut(
                    id=n.id,
                    description=n.description,
                    capability=n.capability,
                    difficulty=n.difficulty,
                    is_discriminative=n.is_discriminative,
                )
                for n in dag.nodes
            ],
            critical_path=dag.critical_path,
            estimated_difficulty=dag.estimated_difficulty,
        )

    # Execution + Scoring per agent
    rankings: list[RankedAgentOut] = []
    agent_details: list[AgentDetailOut] = []

    for result in agent_results:
        agent = result.agent

        # Execute probes via probe-runner
        exec_results: list[ProbeExecutionResult] = []
        if result.validated_plan and result.validated_plan.probes:
            remotes = [{"type": r.type, "url": r.url} for r in agent.remotes]
            try:
                exec_results = await _execute_probes(
                    result.validated_plan, agent.agent_id, remotes,
                )
            except Exception as e:
                logger.error("Probe execution failed for %s: %s", agent.agent_id, e)
                result.error_code = "EXECUTION_FAILED"
                result.error_detail = str(e)

        # Score
        ranked = await loop.run_in_executor(
            None, _pipeline.score_agent_results, result, exec_results
        )

        ranked_out = RankedAgentOut(
            agent_id=ranked.agent_id,
            theta=ranked.theta,
            sigma=ranked.sigma,
            confidence=ranked.confidence,
            testability_tier=ranked.testability_tier,
            probe_summary=ranked.probe_summary,
            prior_influence=ranked.prior_influence,
        )
        rankings.append(ranked_out)

        agent_details.append(AgentDetailOut(
            agent_id=agent.agent_id,
            ranked=ranked_out,
            error_code=result.error_code,
            error_detail=result.error_detail,
            timings=result.timings,
            probes_generated=len(result.probe_plan.probes) if result.probe_plan else 0,
            probes_validated=len(result.validated_plan.probes) if result.validated_plan else 0,
            probes_executed=len(exec_results),
        ))

    # Sort rankings by theta descending
    rankings.sort(key=lambda r: r.theta, reverse=True)

    # Per-request token metrics
    request_calls = _llm.call_log[log_start:]
    input_tokens = sum(r.input_tokens for r in request_calls)
    output_tokens = sum(r.output_tokens for r in request_calls)

    total_time_ms = int((time.monotonic() - t_start) * 1000)

    return ProbeResponse(
        rankings=rankings,
        task_dag=task_dag_out,
        agent_details=agent_details,
        total_time_ms=total_time_ms,
        llm_calls=len(request_calls),
        token_usage=TokenUsage(
            input=input_tokens,
            output=output_tokens,
            total=input_tokens + output_tokens,
        ),
    )
