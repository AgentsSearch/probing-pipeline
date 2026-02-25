#!/usr/bin/env python3
"""Run the probing pipeline against real MCP server schemas with live LLM.

Calls Cerebras Qwen3-235B (or any OpenAI-compatible endpoint) for every stage.
FAISS index is built from real tool schemas of 7 MCP servers.
Execution is simulated (no Stream C sandbox) — all probes pass.

Usage:
    export CEREBRAS_API_KEY=your_key_here
    python scripts/run_real_scenarios.py

    # Single scenario:
    python scripts/run_real_scenarios.py -s database_introspection

    # Different provider:
    export OPENAI_API_KEY=your_key_here
    python scripts/run_real_scenarios.py \
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
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.llm.client import LLMClient
from src.models.integration import (
    ActionStep,
    CandidateAgent,
    ProbeExecutionResult,
    RetrievalResult,
)
from src.pipeline import PipelineConfig, ProbePipeline
from src.tool_index.indexer import ToolIndexer, ToolRecord
from src.tool_index.retriever import ToolRetriever

# ---------------------------------------------------------------------------
# Logging — suppress library noise, keep pipeline + LLM logs
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)
# Show our pipeline stage logs
for name in ("src.stages.task_analysis", "src.stages.tool_alignment",
             "src.stages.probe_generation", "src.stages.probe_validation",
             "src.scoring.birt", "src.scoring.confidence", "src.scoring.prior",
             "src.pipeline", "src.llm.client", "src.tool_index.retriever"):
    logging.getLogger(name).setLevel(logging.INFO)

logger = logging.getLogger("scenario_runner")
logger.setLevel(logging.INFO)

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# ANSI colours (auto-disabled on non-TTY)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _green(t: str) -> str:
    return _c("32", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _red(t: str) -> str:
    return _c("31", t)


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


def _fmt_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _log_step(label: str, detail: str, elapsed: float) -> None:
    tag = _dim(f"[{_fmt_time(elapsed):>7}]")
    print(f"  {tag} {_bold(label)}: {detail}")


def _section(title: str, char: str = "─", width: int = 64) -> None:
    print(f"\n{char * width}")
    print(f"  {_bold(title)}")
    print(f"{char * width}")


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------


def build_index(servers_path: Path) -> tuple[str, int, list[dict]]:
    """Build FAISS index from real MCP server definitions."""
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
    index_dir = tempfile.mkdtemp(prefix="real_mcp_index_")
    indexer.save(index_dir)
    return index_dir, len(indexer.tools), servers


# ---------------------------------------------------------------------------
# Execution simulation (Stream C not available)
# ---------------------------------------------------------------------------


def simulate_execution(plan: Any, agent_id: str) -> list[ProbeExecutionResult]:
    """Simulate probe execution — all probes pass."""
    results = []
    for probe in plan.probes:
        results.append(ProbeExecutionResult(
            agent_id=agent_id,
            probe_id=probe.probe_id,
            output={"status": "simulated", "tool": probe.tool},
            trajectory=[
                ActionStep(action="call_tool", tool_name=probe.tool, result="ok"),
            ],
            latency_ms=150,
            success=True,
        ))
    return results


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------


def run_scenario(
    scenario: dict,
    retriever: ToolRetriever,
    llm: LLMClient,
    config: PipelineConfig,
) -> list[dict]:
    """Run one scenario end-to-end with live LLM calls."""
    query = scenario["query"]

    _section(f"SCENARIO: {scenario['id']}", "=")
    print(f"  {_dim('Query:')}   {query}")
    print(f"  {_dim('Domain:')}  {scenario['expected_domain']}")
    print(f"  {_dim('Expect:')}  best={_green(scenario['best_candidate'])}, "
          f"tools={scenario['relevant_tools']}")

    candidates = [CandidateAgent(**c) for c in scenario["candidates"]]
    print(f"  {_dim('Agents:')}  {', '.join(c.agent_id for c in candidates)}")

    pipeline = ProbePipeline(llm=llm, retriever=retriever, config=config)

    # Track LLM calls for this scenario
    calls_before = len(llm.call_log)

    # ─── Stage 1: Task Analysis (shared across agents) ───
    _section("Stage 1: Task Analysis", "─", 50)

    dag, stage1_time = pipeline.run_stage_1(query)

    if dag is None:
        _log_step(_red("FAILED"), "Could not decompose query", stage1_time)
        return []

    _log_step(
        _green("Decomposed"),
        f"{len(dag.nodes)} subtasks, difficulty={dag.estimated_difficulty:.2f}, "
        f"intent={dag.intent}, domain={dag.domain}",
        stage1_time,
    )
    for node in dag.nodes:
        disc = _yellow(" [DISC]") if node.is_discriminative else ""
        print(f"           {node.id}: {node.description} "
              f"{_dim(f'(d={node.difficulty:.2f}, cap={node.capability})')}{disc}")
    print(f"           {_dim('Critical path:')} {' -> '.join(dag.critical_path)}")
    print(f"           {_dim('Eval dims:')} {', '.join(dag.evaluation_dimensions)}")

    # ─── Stages 2-4 + Scoring per agent ───
    rankings = []

    for agent in candidates:
        _section(f"Agent: {agent.agent_id}", "─", 50)
        print(f"  {_dim('retrieval_score=')}  {agent.retrieval_score:.2f}"
              f"  {_dim('elo=')}  {agent.arena_elo}"
              f"  {_dim('rating=')}  {agent.community_rating}")

        agent_calls_before = len(llm.call_log)

        # Stages 2-4
        t_agent_start = time.monotonic()
        result = pipeline.run_stages_2_to_4_for_agent(dag, agent)
        t_agent_total = time.monotonic() - t_agent_start

        agent_calls = len(llm.call_log) - agent_calls_before

        # --- Stage 2 ---
        t2 = result.timings.get("stage_2_alignment", 0)
        if result.alignment:
            matched = len(result.alignment.alignments)
            unmatched = len(result.alignment.unmatched_subtasks)
            cov = result.alignment.coverage_score
            color = _green if cov > 0.5 else _yellow if cov > 0 else _red
            _log_step("Stage 2 Align", f"coverage={color(f'{cov:.2f}')}, "
                      f"{matched} matched, {unmatched} unmatched", t2)
            for a in result.alignment.alignments:
                print(f"           {a.subtask_id} -> {_bold(a.tool_name)} "
                      f"{_dim(f'({a.match_type}, conf={a.confidence:.2f}, '
                              f'rerank={a.rerank_score:.2f})')}")
            if result.alignment.unmatched_subtasks:
                print(f"           {_dim('unmatched:')} "
                      f"{', '.join(result.alignment.unmatched_subtasks)}")
        elif result.error_code:
            _log_step("Stage 2 Align", _red(f"FAILED — {result.error_code}: "
                                             f"{result.error_detail}"), t2)
            ranked = pipeline.score_agent_results(result, [])
            rankings.append(ranked)
            continue

        # --- Stage 3 ---
        t3 = result.timings.get("stage_3_generation", 0)
        if result.probe_plan:
            n = len(result.probe_plan.probes)
            tools = ", ".join(p.tool for p in result.probe_plan.probes) or "none"
            _log_step("Stage 3 Gen", f"{n} probe(s) -> [{tools}]", t3)
        elif result.error_code:
            _log_step("Stage 3 Gen", _red(f"FAILED — {result.error_code}: "
                                           f"{result.error_detail}"), t3)
            ranked = pipeline.score_agent_results(result, [])
            rankings.append(ranked)
            continue

        # --- Stage 4 ---
        t4 = result.timings.get("stage_4_validation", 0)
        if result.validated_plan:
            vcount = len(result.validated_plan.probes)
            ocount = len(result.probe_plan.probes) if result.probe_plan else 0
            _log_step("Stage 4 Valid", f"{vcount}/{ocount} probes passed", t4)
            for probe in result.validated_plan.probes:
                print(f"           {probe.probe_id}: {_bold(probe.tool)} "
                      f"{_dim(f'd={probe.estimated_difficulty:.2f} '
                              f'a={probe.discrimination:.2f}')} "
                      f"[{probe.priority}]")
                args_str = json.dumps(probe.arguments, ensure_ascii=False)
                if len(args_str) > 100:
                    args_str = args_str[:97] + "..."
                print(f"             {_dim('args=')} {args_str}")
                rubric_names = ", ".join(f"{r.name}({r.weight:.0%})" for r in probe.rubric)
                print(f"             {_dim('rubric=')} {rubric_names}")
        elif result.error_code:
            _log_step("Stage 4 Valid", _red(f"FAILED — {result.error_code}: "
                                             f"{result.error_detail}"), t4)
            ranked = pipeline.score_agent_results(result, [])
            rankings.append(ranked)
            continue

        # --- Execution (simulated — all pass) ---
        t0 = time.monotonic()
        if result.validated_plan and result.validated_plan.probes:
            exec_results = simulate_execution(result.validated_plan, agent.agent_id)
            _log_step("Execution",
                      f"{_green(f'{len(exec_results)} pass')} {_dim('(simulated)')}",
                      time.monotonic() - t0)
        else:
            exec_results = []
            _log_step("Execution", _dim("skipped — no valid probes"), time.monotonic() - t0)

        # --- Scoring ---
        t0 = time.monotonic()
        ranked = pipeline.score_agent_results(result, exec_results)
        tier_color = (_green if ranked.testability_tier == "FULLY_PROBED"
                      else _yellow if ranked.testability_tier == "PARTIALLY_PROBED"
                      else _red)
        _log_step(
            "Scoring",
            f"theta={ranked.theta:.3f}  sigma={ranked.sigma:.3f}  "
            f"conf={ranked.confidence:.3f}  {tier_color(ranked.testability_tier)}  "
            f"prior_influence={ranked.prior_influence:.2f}",
            time.monotonic() - t0,
        )

        _log_step("Agent total",
                   f"{agent.agent_id} ({agent_calls} LLM calls)", t_agent_total)
        rankings.append(ranked)

    # ─── Rankings ───
    rankings.sort(key=lambda r: r.theta, reverse=True)

    scenario_calls = len(llm.call_log) - calls_before

    _section("Rankings", "─", 50)
    print(f"  {'#':<3} {'Agent':<24} {'Score':>6} {'±':>6} {'Conf':>6}  "
          f"{'Tier':<18} Probes")
    print(f"  {'─' * 90}")

    for i, r in enumerate(rankings, 1):
        is_best = r.agent_id == scenario["best_candidate"]
        marker = _green("  ✓") if is_best else "   "
        print(f"  {i:<3} {r.agent_id:<24} {r.theta:>6.3f} {r.sigma:>6.3f} "
              f"{r.confidence:>6.3f}  {r.testability_tier:<18} "
              f"{r.probe_summary}{marker}")

    print(f"\n  {_dim(f'LLM calls this scenario: {scenario_calls}')}")

    return [
        {"rank": i, "agent_id": r.agent_id, "theta": r.theta, "sigma": r.sigma,
         "confidence": r.confidence, "tier": r.testability_tier}
        for i, r in enumerate(rankings, 1)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run probing pipeline with live LLM against real MCP server schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  github_repo_search      Find Python ML repos on GitHub (3 agents)
  web_search_save         Search web for AI news + save to file (3 agents)
  database_introspection  List tables, describe schema, count users (2 agents)
  web_scraping            Scrape page, screenshot, extract data (3 agents)
  slack_notification      Send deployment notification to Slack (3 agents)
        """,
    )
    parser.add_argument(
        "--scenario", "-s", default=None,
        help="Scenario ID to run (default: all)",
    )
    parser.add_argument(
        "--api-key-env", default="CEREBRAS_API_KEY",
        help="Env var for API key (default: CEREBRAS_API_KEY)",
    )
    parser.add_argument("--base-url", default=None, help="Override LLM base URL")
    parser.add_argument("--model", default=None, help="Override LLM model name")
    parser.add_argument("--budget", type=int, default=2, help="Probe budget per agent")
    args = parser.parse_args()

    total_start = time.monotonic()

    # ─── Header ───
    print(f"\n{'=' * 64}")
    print(f"  {_bold('PROBING PIPELINE — Real MCP Scenarios')}")
    print(f"{'=' * 64}")
    print(f"  {_dim('Budget:')} {args.budget} probes/agent")

    # ─── LLM Client ───
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"\n  {_red('Error:')} Set {args.api_key_env}")
        print(f"    export {args.api_key_env}=your_key_here\n")
        sys.exit(1)

    llm = LLMClient.from_config(api_key=api_key)
    if args.base_url:
        llm.base_url = args.base_url
        llm.__post_init__()
    if args.model:
        llm.model = args.model

    print(f"  {_dim('Model:')}  {llm.model}")
    print(f"  {_dim('URL:')}    {llm.base_url}")

    # ─── Build FAISS Index ───
    _section("Setup: Building FAISS Index", "─", 50)
    t0 = time.monotonic()
    servers_path = FIXTURES / "real_mcp_servers.json"
    index_dir, tool_count, servers = build_index(servers_path)
    retriever = ToolRetriever(index_dir)
    _log_step("Index built", f"{tool_count} tools across {len(servers)} servers",
              time.monotonic() - t0)

    server_summary = ", ".join(f"{s['server_id']}({len(s['tools'])})" for s in servers)
    print(f"           {_dim(server_summary)}")

    # ─── Load scenarios ───
    with open(FIXTURES / "real_queries.json") as f:
        all_scenarios = json.load(f)

    if args.scenario:
        scenarios = [s for s in all_scenarios if s["id"] == args.scenario]
        if not scenarios:
            valid = ", ".join(s["id"] for s in all_scenarios)
            print(f"\n  {_red('Error:')} Unknown scenario '{args.scenario}'")
            print(f"  Valid: {valid}\n")
            sys.exit(1)
    else:
        scenarios = all_scenarios

    print(f"  {_dim('Scenarios:')} {len(scenarios)} to run")

    # ─── Run ───
    config = PipelineConfig(probe_budget=args.budget)
    all_rankings = {}

    for scenario in scenarios:
        rankings = run_scenario(scenario, retriever, llm, config)
        all_rankings[scenario["id"]] = rankings

    # ─── Grand Summary ───
    total_time = time.monotonic() - total_start
    tokens = llm.total_tokens()

    print(f"\n{'=' * 64}")
    print(f"  {_bold('SUMMARY')}")
    print(f"{'=' * 64}")

    for sid, ranks in all_rankings.items():
        scenario = next(s for s in all_scenarios if s["id"] == sid)
        expected = scenario["best_candidate"]
        actual = ranks[0]["agent_id"] if ranks else "N/A"
        correct = actual == expected
        icon = _green("PASS") if correct else _red("FAIL")
        print(f"  [{icon}] {sid:<30} expected={expected:<22} got={actual}")

    passed = sum(
        1 for sid, ranks in all_rankings.items()
        if ranks and ranks[0]["agent_id"] == next(
            s["best_candidate"] for s in all_scenarios if s["id"] == sid
        )
    )
    total = len(all_rankings)
    pct = _green if passed == total else _yellow if passed > 0 else _red

    print(f"\n  {_dim('Scenarios:')}  {pct(f'{passed}/{total}')} ranked correctly")
    print(f"  {_dim('Total time:')} {_fmt_time(total_time)}")
    print(f"  {_dim('LLM calls:')} {len(llm.call_log)}")
    print(f"  {_dim('Tokens:')}    {tokens['input']:,} in + {tokens['output']:,} out "
          f"= {tokens['total']:,} total")

    # Per-call breakdown
    print(f"\n  {_dim('Call log:')}")
    for i, call in enumerate(llm.call_log, 1):
        status = _green("✓") if call.success else _red("✗")
        print(f"    {status} {i:>2}. {_fmt_time(call.latency_ms / 1000):>7}  "
              f"{call.input_tokens:>5} in  {call.output_tokens:>5} out  "
              f"{_dim(call.prompt_hash)}")

    print()


if __name__ == "__main__":
    main()
