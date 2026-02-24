"""Integration test — runs the full pipeline with mock LLM and FAISS index.

Tests the end-to-end flow: query -> TaskDAG -> alignment -> probes -> validation -> scoring.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.models.integration import (
    ActionStep,
    CandidateAgent,
    ProbeExecutionResult,
    RetrievalResult,
)
from src.pipeline import ProbePipeline, PipelineConfig
from src.tool_index.indexer import ToolIndexer, ToolRecord

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load_servers():
    with open(FIXTURES / "sample_mcp_servers.json") as f:
        return json.load(f)


def _build_tool_index(tmpdir: str) -> str:
    """Build a FAISS index from fixture MCP servers and return the index dir."""
    servers = _load_servers()
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
    indexer.save(tmpdir)
    return tmpdir


def _mock_task_analysis_response():
    """Mock LLM response for Stage 1."""
    return {
        "query": "Find the current weather in London and format it as a markdown summary",
        "intent": "data_retrieval",
        "domain": "general",
        "nodes": [
            {
                "id": "S1",
                "description": "Retrieve current weather data for London",
                "capability": "api_call",
                "difficulty": 0.3,
                "is_discriminative": True,
                "depends_on": [],
            },
            {
                "id": "S2",
                "description": "Format weather data as markdown summary",
                "capability": "text_formatting",
                "difficulty": 0.2,
                "is_discriminative": False,
                "depends_on": ["S1"],
            },
        ],
        "edges": [["S1", "S2"]],
        "critical_path": ["S1", "S2"],
        "estimated_difficulty": 0.3,
        "evaluation_dimensions": ["correctness", "completeness", "formatting"],
    }


def _mock_alignment_response():
    """Mock LLM response for Stage 2 reranker."""
    return {
        "alignments": [
            {
                "subtask_id": "S1",
                "tool_name": "get_current_weather",
                "server_id": "weather-server",
                "match_type": "direct",
                "confidence": 0.95,
                "rerank_score": 0.95,
                "parameter_mapping": {
                    "city": {
                        "subtask_param": "location",
                        "tool_param": "city",
                        "transform": None,
                    }
                },
            }
        ]
    }


def _mock_probe_generation_response():
    """Mock LLM response for Stage 3."""
    return {
        "probe_id": "P1",
        "targets_subtask": "S1",
        "tool": "get_current_weather",
        "arguments": {"city": "London", "units": "celsius"},
        "estimated_difficulty": 0.3,
        "discrimination": 1.2,
        "rubric": [
            {
                "name": "correctness",
                "weight": 0.5,
                "criteria": "Returns valid weather data for London",
                "pass_threshold": "Temperature, humidity, and condition fields present and plausible",
            },
            {
                "name": "completeness",
                "weight": 0.3,
                "criteria": "All requested weather fields are populated",
                "pass_threshold": "At least temperature and condition are present",
            },
            {
                "name": "latency",
                "weight": 0.2,
                "criteria": "Response returned within timeout",
                "pass_threshold": "Response received in under 10 seconds",
            },
        ],
        "timeout_seconds": 15,
        "priority": "PRIMARY",
    }


class TestFullPipeline:
    """Integration test running Stages 1-4 + scoring with mocked LLM."""

    def test_stages_1_to_4(self):
        """Test the full pipeline from query to validated probe plan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = _build_tool_index(tmpdir)

            from src.tool_index.retriever import ToolRetriever
            retriever = ToolRetriever(index_dir)

            # Mock LLM to return stage-appropriate responses
            mock_llm = MagicMock()
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _mock_task_analysis_response()
                elif call_count == 2:
                    return _mock_alignment_response()
                else:
                    return _mock_probe_generation_response()

            mock_llm.complete_json.side_effect = side_effect

            pipeline = ProbePipeline(
                llm=mock_llm,
                retriever=retriever,
                config=PipelineConfig(probe_budget=2, total_timeout=30),
            )

            retrieval_result = RetrievalResult(
                query="Find the current weather in London and format it as a markdown summary",
                candidates=[
                    CandidateAgent(
                        agent_id="weather-server",
                        retrieval_score=0.85,
                        mcp_server_url="http://localhost:8001",
                        arena_elo=1400,
                        community_rating=4.2,
                    ),
                ],
            )

            dag, agent_results = pipeline.run_stages_1_to_4(retrieval_result)

            # Stage 1 assertions
            assert dag is not None
            assert len(dag.nodes) == 2
            assert dag.intent == "data_retrieval"

            # Stage 2-4 assertions
            assert len(agent_results) == 1
            result = agent_results[0]
            assert result.error_code is None
            assert result.alignment is not None
            assert result.validated_plan is not None

    def test_scoring_after_execution(self):
        """Test scoring with mock execution results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = _build_tool_index(tmpdir)

            from src.tool_index.retriever import ToolRetriever
            retriever = ToolRetriever(index_dir)

            mock_llm = MagicMock()
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _mock_task_analysis_response()
                elif call_count == 2:
                    return _mock_alignment_response()
                else:
                    return _mock_probe_generation_response()

            mock_llm.complete_json.side_effect = side_effect

            pipeline = ProbePipeline(
                llm=mock_llm,
                retriever=retriever,
                config=PipelineConfig(probe_budget=2),
            )

            retrieval_result = RetrievalResult(
                query="Find the current weather in London",
                candidates=[
                    CandidateAgent(
                        agent_id="weather-server",
                        retrieval_score=0.85,
                        mcp_server_url="http://localhost:8001",
                        arena_elo=1400,
                        community_rating=4.2,
                    ),
                ],
            )

            dag, agent_results = pipeline.run_stages_1_to_4(retrieval_result)
            result = agent_results[0]

            # Simulate execution results from Stream C
            exec_results = []
            if result.validated_plan:
                for probe in result.validated_plan.probes:
                    exec_results.append(
                        ProbeExecutionResult(
                            agent_id="weather-server",
                            probe_id=probe.probe_id,
                            output={"temperature": 15.2, "humidity": 72, "condition": "cloudy"},
                            trajectory=[
                                ActionStep(action="call_tool", tool_name="get_current_weather", result="ok"),
                            ],
                            latency_ms=230,
                            success=True,
                        )
                    )

            ranked = pipeline.score_agent_results(result, exec_results)

            assert ranked.agent_id == "weather-server"
            assert 0.0 <= ranked.theta <= 1.0
            assert 0.0 <= ranked.confidence <= 1.0
            assert ranked.testability_tier in ("FULLY_PROBED", "PARTIALLY_PROBED", "UNTESTABLE")
            assert "PASS" in ranked.probe_summary or "No probes" in ranked.probe_summary

    def test_pipeline_with_no_matching_tools(self):
        """Test graceful handling when agent has no matching tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = _build_tool_index(tmpdir)

            from src.tool_index.retriever import ToolRetriever
            retriever = ToolRetriever(index_dir)

            mock_llm = MagicMock()
            call_count = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _mock_task_analysis_response()
                else:
                    # Return no alignments
                    return {"alignments": []}

            mock_llm.complete_json.side_effect = side_effect

            pipeline = ProbePipeline(llm=mock_llm, retriever=retriever)

            retrieval_result = RetrievalResult(
                query="Find weather in London",
                candidates=[
                    CandidateAgent(
                        agent_id="nonexistent-server",
                        retrieval_score=0.3,
                        mcp_server_url="http://localhost:9999",
                    ),
                ],
            )

            dag, agent_results = pipeline.run_stages_1_to_4(retrieval_result)
            assert dag is not None
            result = agent_results[0]

            # Should still complete without crashing
            assert result.error_code is None or result.error_code in (
                "ALIGNMENT_FAILED", "PROBE_GENERATION_FAILED"
            )

            # Score with no execution
            ranked = pipeline.score_agent_results(result, [])
            assert ranked.testability_tier in ("UNTESTABLE", "PARTIALLY_PROBED")
