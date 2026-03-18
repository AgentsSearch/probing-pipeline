"""Unit tests for data models."""

import pytest

from src.models.task import SubtaskNode, TaskDAG
from src.models.alignment import ToolAlignment, AlignmentMap, ParameterMap
from src.models.probe import RubricDimension, Probe, ProbePlan, ProbeTemplate
from src.models.scoring import GaussianPrior, PosteriorEstimate
from src.models.integration import (
    CandidateAgent,
    InlineTool,
    LLMExtracted,
    RemoteEndpoint,
    RetrievalResult,
    ProbeExecutionRequest,
    ProbeExecutionResult,
    ActionStep,
    RankedAgent,
)


# --- SubtaskNode ---


class TestSubtaskNode:
    def test_valid_creation(self):
        node = SubtaskNode(
            id="S1", description="test", capability="file_read",
            difficulty=0.5, is_discriminative=True,
        )
        assert node.id == "S1"
        assert node.difficulty == 0.5

    def test_difficulty_below_zero_raises(self):
        with pytest.raises(ValueError, match="difficulty must be in"):
            SubtaskNode(
                id="S1", description="test", capability="x",
                difficulty=-0.1, is_discriminative=False,
            )

    def test_difficulty_above_one_raises(self):
        with pytest.raises(ValueError, match="difficulty must be in"):
            SubtaskNode(
                id="S1", description="test", capability="x",
                difficulty=1.1, is_discriminative=False,
            )

    def test_default_depends_on(self):
        node = SubtaskNode(
            id="S1", description="test", capability="x",
            difficulty=0.5, is_discriminative=False,
        )
        assert node.depends_on == []


# --- TaskDAG ---


class TestTaskDAG:
    def _make_nodes(self, n: int) -> list[SubtaskNode]:
        return [
            SubtaskNode(
                id=f"S{i}", description=f"task {i}", capability="general",
                difficulty=0.3, is_discriminative=(i % 2 == 0),
            )
            for i in range(1, n + 1)
        ]

    def test_valid_creation(self):
        nodes = self._make_nodes(3)
        dag = TaskDAG(
            query="test", intent="test", domain="test",
            nodes=nodes, edges=[("S1", "S2")],
            critical_path=["S1", "S2"], estimated_difficulty=0.4,
        )
        assert len(dag.nodes) == 3

    def test_max_six_nodes(self):
        with pytest.raises(ValueError, match="Maximum 6"):
            TaskDAG(
                query="test", intent="test", domain="test",
                nodes=self._make_nodes(7), edges=[],
                critical_path=[], estimated_difficulty=0.5,
            )

    def test_difficulty_validation(self):
        with pytest.raises(ValueError, match="estimated_difficulty"):
            TaskDAG(
                query="test", intent="test", domain="test",
                nodes=self._make_nodes(2), edges=[],
                critical_path=[], estimated_difficulty=1.5,
            )

    def test_get_node(self):
        nodes = self._make_nodes(3)
        dag = TaskDAG(
            query="test", intent="test", domain="test",
            nodes=nodes, edges=[], critical_path=[], estimated_difficulty=0.5,
        )
        assert dag.get_node("S2") is nodes[1]
        assert dag.get_node("S99") is None

    def test_discriminative_nodes(self):
        nodes = self._make_nodes(4)
        dag = TaskDAG(
            query="test", intent="test", domain="test",
            nodes=nodes, edges=[], critical_path=[], estimated_difficulty=0.5,
        )
        disc = dag.discriminative_nodes()
        assert all(n.is_discriminative for n in disc)
        assert len(disc) == 2  # S2, S4


# --- ToolAlignment / AlignmentMap ---


class TestAlignment:
    def test_confidence_validation(self):
        with pytest.raises(ValueError, match="confidence"):
            ToolAlignment(
                subtask_id="S1", tool_name="t", server_id="s",
                match_type="direct", confidence=1.5,
                retrieval_score=0.8, rerank_score=0.9,
            )

    def test_coverage_validation(self):
        with pytest.raises(ValueError, match="coverage_score"):
            AlignmentMap(
                agent_id="a", server_tool_count=10, tools_evaluated=5,
                alignments=[], coverage_score=-0.1,
            )

    def test_best_alignment_for_subtask(self):
        a1 = ToolAlignment(
            subtask_id="S1", tool_name="t1", server_id="s",
            match_type="direct", confidence=0.7,
            retrieval_score=0.8, rerank_score=0.7,
        )
        a2 = ToolAlignment(
            subtask_id="S1", tool_name="t2", server_id="s",
            match_type="partial", confidence=0.9,
            retrieval_score=0.6, rerank_score=0.9,
        )
        amap = AlignmentMap(
            agent_id="a", server_tool_count=10, tools_evaluated=2,
            alignments=[a1, a2], coverage_score=1.0,
        )
        best = amap.best_alignment_for_subtask("S1")
        assert best.tool_name == "t2"
        assert amap.best_alignment_for_subtask("S99") is None


# --- Probe ---


class TestProbe:
    def _make_rubric(self, n: int = 2) -> list[RubricDimension]:
        return [
            RubricDimension(
                name=f"dim{i}", weight=1.0 / n,
                criteria="test", pass_threshold="test",
            )
            for i in range(n)
        ]

    def test_valid_probe(self):
        probe = Probe(
            probe_id="P1", targets_subtask="S1", tool="t",
            arguments={"x": 1}, estimated_difficulty=0.5,
            discrimination=1.0, rubric=self._make_rubric(2),
            timeout_seconds=15, priority="PRIMARY",
        )
        assert probe.probe_id == "P1"

    def test_min_rubric_dimensions(self):
        with pytest.raises(ValueError, match="at least 2"):
            Probe(
                probe_id="P1", targets_subtask="S1", tool="t",
                arguments={"x": 1}, estimated_difficulty=0.5,
                discrimination=1.0, rubric=self._make_rubric(1),
                timeout_seconds=15, priority="PRIMARY",
            )

    def test_difficulty_validation(self):
        with pytest.raises(ValueError, match="estimated_difficulty"):
            Probe(
                probe_id="P1", targets_subtask="S1", tool="t",
                arguments={"x": 1}, estimated_difficulty=2.0,
                discrimination=1.0, rubric=self._make_rubric(2),
                timeout_seconds=15, priority="PRIMARY",
            )

    def test_rubric_weight_validation(self):
        with pytest.raises(ValueError, match="weight"):
            RubricDimension(name="x", weight=1.5, criteria="c", pass_threshold="p")


# --- Scoring Models ---


class TestScoringModels:
    def test_gaussian_prior(self):
        p = GaussianPrior(mu=0.5, sigma=0.3)
        assert p.mu == 0.5
        assert p.sigma == 0.3

    def test_gaussian_prior_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            GaussianPrior(mu=0.5, sigma=-0.1)

    def test_posterior_estimate(self):
        est = PosteriorEstimate(
            theta=0.7, sigma=0.2, confidence=0.8,
            n_probes=2, testability_tier="FULLY_PROBED",
            prior_influence=0.3,
        )
        assert est.testability_tier == "FULLY_PROBED"

    def test_posterior_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            PosteriorEstimate(
                theta=0.5, sigma=-0.1, confidence=0.5,
                n_probes=1, testability_tier="PARTIALLY_PROBED",
                prior_influence=0.5,
            )


# --- Integration Models ---


class TestIntegrationModels:
    def test_candidate_agent(self):
        agent = CandidateAgent(
            agent_id="a1", retrieval_score=0.8,
            mcp_server_url="http://localhost:8080",
            arena_elo=1500, community_rating=4.0,
        )
        assert agent.arena_elo == 1500

    def test_candidate_agent_optional_fields(self):
        agent = CandidateAgent(
            agent_id="a1", retrieval_score=0.8,
            mcp_server_url="http://localhost:8080",
        )
        assert agent.arena_elo is None
        assert agent.community_rating is None

    def test_retrieval_result(self):
        rr = RetrievalResult(query="test", candidates=[])
        assert rr.candidates == []

    def test_action_step_defaults(self):
        step = ActionStep(action="call_tool")
        assert step.tool_name is None
        assert step.latency_ms == 0

    def test_ranked_agent(self):
        ra = RankedAgent(
            agent_id="a1", theta=0.7, sigma=0.2,
            confidence=0.8, testability_tier="FULLY_PROBED",
            probe_summary="P1: PASS", prior_influence=0.3,
        )
        assert ra.probe_summary == "P1: PASS"


# --- New Rich Schema Models ---


class TestRemoteEndpoint:
    def test_creation(self):
        ep = RemoteEndpoint(type="streamable-http", url="https://api.example.com/mcp")
        assert ep.type == "streamable-http"
        assert ep.url == "https://api.example.com/mcp"

    def test_sse_type(self):
        ep = RemoteEndpoint(type="sse", url="http://localhost:8080/sse")
        assert ep.type == "sse"


class TestLLMExtracted:
    def test_defaults(self):
        llm = LLMExtracted()
        assert llm.capabilities == []
        assert llm.limitations == []
        assert llm.requirements == []

    def test_with_data(self):
        llm = LLMExtracted(
            capabilities=["file read", "file write"],
            limitations=["no binary files"],
            requirements=["filesystem access"],
        )
        assert len(llm.capabilities) == 2
        assert "no binary files" in llm.limitations


class TestInlineTool:
    def test_creation(self):
        tool = InlineTool(
            name="get_weather",
            description="Get current weather",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert tool.name == "get_weather"
        assert "city" in tool.input_schema["properties"]

    def test_defaults(self):
        tool = InlineTool(name="ping", description="Health check")
        assert tool.input_schema == {}


class TestCandidateAgentRichSchema:
    def test_new_score_field(self):
        agent = CandidateAgent(agent_id="a1", score=0.9)
        assert agent.score == 0.9
        assert agent.retrieval_score == 0.9  # alias

    def test_legacy_retrieval_score(self):
        agent = CandidateAgent(
            agent_id="a1", retrieval_score=0.85,
            mcp_server_url="http://localhost:8080",
        )
        assert agent.score == 0.85
        assert agent.retrieval_score == 0.85

    def test_legacy_mcp_server_url(self):
        agent = CandidateAgent(
            agent_id="a1", score=0.7,
            mcp_server_url="http://localhost:8080",
        )
        assert agent.mcp_server_url == "http://localhost:8080"
        assert len(agent.remotes) == 1
        assert agent.remotes[0].type == "sse"

    def test_remotes_with_preference(self):
        agent = CandidateAgent(
            agent_id="a1", score=0.7,
            remotes=[
                RemoteEndpoint(type="sse", url="http://localhost:8080/sse"),
                RemoteEndpoint(type="streamable-http", url="http://localhost:8080/mcp"),
            ],
        )
        assert agent.best_remote_url() == "http://localhost:8080/mcp"
        assert agent.mcp_server_url == "http://localhost:8080/mcp"

    def test_no_remotes(self):
        agent = CandidateAgent(agent_id="a1", score=0.5)
        assert agent.best_remote_url() is None
        assert agent.mcp_server_url is None

    def test_missing_score_raises(self):
        with pytest.raises(TypeError, match="score.*required"):
            CandidateAgent(agent_id="a1")

    def test_rich_fields(self):
        agent = CandidateAgent(
            agent_id="a1",
            score=0.9,
            description="A weather agent",
            tools=[InlineTool(name="get_weather", description="Get weather")],
            llm_extracted=LLMExtracted(
                capabilities=["weather lookup"],
                limitations=["no historical data"],
            ),
            documentation_quality=0.85,
            is_available=True,
            testability_tier="FULLY_PROBED",
        )
        assert agent.description == "A weather agent"
        assert len(agent.tools) == 1
        assert agent.llm_extracted.limitations == ["no historical data"]
        assert agent.documentation_quality == 0.85
        assert agent.is_available is True
        assert agent.testability_tier == "FULLY_PROBED"

    def test_is_available_default(self):
        agent = CandidateAgent(agent_id="a1", score=0.5)
        assert agent.is_available is True

    def test_inline_tools_default(self):
        agent = CandidateAgent(agent_id="a1", score=0.5)
        assert agent.tools == []


class TestProbeExecutionRequestRemotes:
    def test_with_remotes(self):
        req = ProbeExecutionRequest(
            agent_id="a1",
            probes=[],
            total_timeout=30,
            remotes=[RemoteEndpoint(type="sse", url="http://localhost:8080")],
        )
        assert len(req.remotes) == 1

    def test_legacy_mcp_server_url(self):
        req = ProbeExecutionRequest(
            agent_id="a1",
            probes=[],
            total_timeout=30,
            mcp_server_url="http://localhost:8080",
        )
        assert req.mcp_server_url == "http://localhost:8080"
