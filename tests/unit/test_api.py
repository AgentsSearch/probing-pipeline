"""Tests for the FastAPI probing pipeline API."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    CandidateAgentIn,
    ProbeRequest,
    to_candidate_agent,
)
from src.models.integration import CandidateAgent


# ── Schema validation tests ──


class TestProbeRequestValidation:
    """Test Pydantic validation on ProbeRequest."""

    def test_missing_query(self):
        with pytest.raises(ValidationError):
            ProbeRequest(query="", candidates=[{"agent_id": "a", "score": 0.5}])

    def test_empty_candidates(self):
        with pytest.raises(ValidationError):
            ProbeRequest(query="test query", candidates=[])

    def test_score_out_of_range_high(self):
        with pytest.raises(ValidationError):
            CandidateAgentIn(agent_id="a", score=1.5)

    def test_score_out_of_range_low(self):
        with pytest.raises(ValidationError):
            CandidateAgentIn(agent_id="a", score=-0.1)

    def test_valid_minimal_request(self):
        req = ProbeRequest(
            query="test query",
            candidates=[CandidateAgentIn(agent_id="a", score=0.5)],
        )
        assert req.query == "test query"
        assert len(req.candidates) == 1

    def test_valid_full_request(self):
        req = ProbeRequest(
            query="Find weather",
            candidates=[
                CandidateAgentIn(
                    agent_id="weather-server",
                    score=0.85,
                    remotes=[{"type": "sse", "url": "http://example.com"}],
                    description="A weather server",
                    tools=[{"name": "get_weather", "description": "Get weather"}],
                    llm_extracted={
                        "capabilities": ["weather lookup"],
                        "limitations": ["no forecasts"],
                        "requirements": [],
                    },
                    documentation_quality=0.8,
                    arena_elo=1200.0,
                    community_rating=4.5,
                ),
            ],
        )
        assert req.candidates[0].description == "A weather server"
        assert len(req.candidates[0].tools) == 1
        assert req.candidates[0].llm_extracted.limitations == ["no forecasts"]


# ── Converter tests ──


class TestToCandidateAgent:
    """Test Pydantic -> dataclass conversion."""

    def test_minimal_conversion(self):
        agent_in = CandidateAgentIn(agent_id="test", score=0.7)
        agent = to_candidate_agent(agent_in)
        assert isinstance(agent, CandidateAgent)
        assert agent.agent_id == "test"
        assert agent.score == 0.7
        assert agent.remotes == []
        assert agent.tools == []

    def test_full_conversion(self):
        agent_in = CandidateAgentIn(
            agent_id="full-agent",
            score=0.9,
            remotes=[{"type": "streamable-http", "url": "http://example.com/mcp"}],
            tools=[{"name": "tool1", "description": "desc", "input_schema": {"type": "object"}}],
            llm_extracted={
                "capabilities": ["cap1"],
                "limitations": ["lim1"],
                "requirements": ["req1"],
            },
            documentation_quality=0.75,
            is_available=True,
            testability_tier="FULLY_TESTABLE",
        )
        agent = to_candidate_agent(agent_in)
        assert agent.remotes[0].type == "streamable-http"
        assert agent.tools[0].name == "tool1"
        assert agent.tools[0].input_schema == {"type": "object"}
        assert agent.llm_extracted.capabilities == ["cap1"]
        assert agent.documentation_quality == 0.75
        assert agent.testability_tier == "FULLY_TESTABLE"

    def test_conversion_preserves_score(self):
        """Ensure score maps correctly (not the deprecated retrieval_score)."""
        agent_in = CandidateAgentIn(agent_id="x", score=0.42)
        agent = to_candidate_agent(agent_in)
        assert agent.score == 0.42
        assert agent.retrieval_score == 0.42  # backward-compat alias


# ── Endpoint tests ──


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline that returns predictable results."""
    from src.models.integration import RankedAgent
    from src.models.task import SubtaskNode, TaskDAG
    from src.pipeline import AgentPipelineResult

    dag = TaskDAG(
        query="test query",
        intent="test",
        domain="general",
        nodes=[
            SubtaskNode(
                id="s1",
                description="Do something",
                capability="general",
                difficulty=0.5,
                is_discriminative=True,
            )
        ],
        edges=[],
        critical_path=["s1"],
        estimated_difficulty=0.5,
    )

    def fake_run_stages(retrieval_result):
        results = []
        for candidate in retrieval_result.candidates:
            if candidate.is_available:
                results.append(AgentPipelineResult(agent=candidate))
            else:
                results.append(AgentPipelineResult(
                    agent=candidate,
                    error_code="ALIGNMENT_FAILED",
                    error_detail="Agent unavailable",
                ))
        return dag, results

    def fake_score(agent_result, exec_results):
        return RankedAgent(
            agent_id=agent_result.agent.agent_id,
            theta=0.6,
            sigma=0.3,
            confidence=0.7,
            testability_tier="FULLY_TESTABLE",
            probe_summary="s1: PASS",
            prior_influence=0.4,
        )

    pipeline = MagicMock()
    pipeline.run_stages_1_to_4 = MagicMock(side_effect=fake_run_stages)
    pipeline.score_agent_results = MagicMock(side_effect=fake_score)
    return pipeline


@pytest.fixture
def mock_llm():
    """Create a mock LLM client with call_log tracking."""
    llm = MagicMock()
    llm.call_log = []
    llm.total_tokens.return_value = {"input": 0, "output": 0, "total": 0}
    return llm


@pytest.fixture
def test_client(mock_pipeline, mock_llm):
    """Create a TestClient with mocked pipeline."""
    from fastapi.testclient import TestClient

    import src.api.app as app_module

    # Patch module-level state
    original_pipeline = app_module._pipeline
    original_llm = app_module._llm
    original_ready = app_module._ready

    app_module._pipeline = mock_pipeline
    app_module._llm = mock_llm
    app_module._ready = True

    client = TestClient(app_module.app)
    yield client

    # Restore
    app_module._pipeline = original_pipeline
    app_module._llm = original_llm
    app_module._ready = original_ready


class TestHealthEndpoint:
    def test_health_when_ready(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "ready": True}

    def test_health_when_not_ready(self):
        from fastapi.testclient import TestClient

        import src.api.app as app_module

        original_ready = app_module._ready
        app_module._ready = False
        client = TestClient(app_module.app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "ready": False}
        app_module._ready = original_ready


class TestProbeEndpoint:
    def test_probe_success(self, test_client):
        response = test_client.post("/probe", json={
            "query": "Find weather in London",
            "candidates": [{"agent_id": "weather-server", "score": 0.85}],
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["rankings"]) == 1
        assert data["rankings"][0]["agent_id"] == "weather-server"
        assert data["rankings"][0]["theta"] == 0.6
        assert data["task_dag"] is not None
        assert data["task_dag"]["query"] == "test query"
        assert len(data["agent_details"]) == 1
        assert data["total_time_ms"] >= 0
        assert data["llm_calls"] == 0
        assert data["token_usage"]["total"] == 0

    def test_probe_multiple_candidates(self, test_client):
        response = test_client.post("/probe", json={
            "query": "Search and summarize",
            "candidates": [
                {"agent_id": "agent-a", "score": 0.9},
                {"agent_id": "agent-b", "score": 0.7},
            ],
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["rankings"]) == 2
        assert len(data["agent_details"]) == 2

    def test_probe_not_ready(self):
        from fastapi.testclient import TestClient

        import src.api.app as app_module

        original_ready = app_module._ready
        app_module._ready = False
        client = TestClient(app_module.app)
        response = client.post("/probe", json={
            "query": "test",
            "candidates": [{"agent_id": "a", "score": 0.5}],
        })
        assert response.status_code == 503
        app_module._ready = original_ready

    def test_probe_validation_error_empty_query(self, test_client):
        response = test_client.post("/probe", json={
            "query": "",
            "candidates": [{"agent_id": "a", "score": 0.5}],
        })
        assert response.status_code == 422

    def test_probe_validation_error_no_candidates(self, test_client):
        response = test_client.post("/probe", json={
            "query": "test",
            "candidates": [],
        })
        assert response.status_code == 422

    def test_probe_validation_error_bad_score(self, test_client):
        response = test_client.post("/probe", json={
            "query": "test",
            "candidates": [{"agent_id": "a", "score": 2.0}],
        })
        assert response.status_code == 422
