"""Integration tests with real MCP server tool schemas.

Tests the full pipeline (Stages 1-4 + scoring) against tool schemas
from 7 real MCP servers: GitHub, Filesystem, Brave Search, SQLite,
Puppeteer, Fetch, and Slack.

LLM calls are mocked with realistic responses derived from each
test scenario, but the FAISS index is built from real tool descriptions.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models.integration import (
    ActionStep,
    CandidateAgent,
    ProbeExecutionResult,
    RetrievalResult,
)
from src.pipeline import PipelineConfig, ProbePipeline
from src.tool_index.indexer import ToolIndexer, ToolRecord
from src.tool_index.retriever import ToolRetriever

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _load_real_servers():
    with open(FIXTURES / "real_mcp_servers.json") as f:
        return json.load(f)


def _load_real_queries():
    with open(FIXTURES / "real_queries.json") as f:
        return json.load(f)


def _build_real_tool_index(tmpdir: str) -> str:
    """Build a FAISS index from all 7 real MCP servers."""
    servers = _load_real_servers()
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


def _get_query_by_id(query_id: str) -> dict:
    queries = _load_real_queries()
    for q in queries:
        if q["id"] == query_id:
            return q
    raise ValueError(f"Query {query_id} not found")


# ---------------------------------------------------------------------------
# Mock LLM response builders
# ---------------------------------------------------------------------------


def _make_alignment_response(
    subtask_id: str,
    tool_name: str,
    server_id: str,
    match_type: str = "direct",
    confidence: float = 0.9,
) -> dict:
    """Build a mock LLM alignment (reranker) response."""
    return {
        "alignments": [
            {
                "subtask_id": subtask_id,
                "tool_name": tool_name,
                "server_id": server_id,
                "match_type": match_type,
                "confidence": confidence,
                "rerank_score": confidence,
                "parameter_mapping": {},
            }
        ]
    }


def _make_probe_response(
    probe_id: str,
    subtask_id: str,
    tool_name: str,
    arguments: dict,
    difficulty: float = 0.3,
    discrimination: float = 1.2,
) -> dict:
    """Build a mock LLM probe generation response."""
    return {
        "probe_id": probe_id,
        "targets_subtask": subtask_id,
        "tool": tool_name,
        "arguments": arguments,
        "estimated_difficulty": difficulty,
        "discrimination": discrimination,
        "rubric": [
            {
                "name": "correctness",
                "weight": 0.5,
                "criteria": "Output matches expected result",
                "pass_threshold": "All required fields present and valid",
            },
            {
                "name": "completeness",
                "weight": 0.3,
                "criteria": "All requested data is returned",
                "pass_threshold": "No missing fields",
            },
            {
                "name": "latency",
                "weight": 0.2,
                "criteria": "Response within timeout",
                "pass_threshold": "Under 10 seconds",
            },
        ],
        "timeout_seconds": 15,
        "priority": "PRIMARY",
    }


def _make_execution_result(
    agent_id: str,
    probe_id: str,
    output: dict,
    success: bool = True,
    latency_ms: int = 200,
) -> ProbeExecutionResult:
    """Build a mock execution result (as if returned by Stream C)."""
    return ProbeExecutionResult(
        agent_id=agent_id,
        probe_id=probe_id,
        output=output,
        trajectory=[
            ActionStep(
                action="call_tool",
                tool_name="test_tool",
                result="ok",
                latency_ms=latency_ms,
            ),
        ],
        latency_ms=latency_ms,
        success=success,
    )


# ---------------------------------------------------------------------------
# Shared FAISS index fixture (built once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_index_dir():
    """Build the FAISS index once for all tests in this module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _build_real_tool_index(tmpdir)
        yield tmpdir


@pytest.fixture()
def retriever(real_index_dir):
    return ToolRetriever(real_index_dir)


# ===========================================================================
# Test 1: GitHub Repo Search — single-agent clear match
# ===========================================================================


class TestGitHubRepoSearch:
    """Query: Find Python ML repos on GitHub and show the README."""

    def test_faiss_retrieves_github_tools(self, retriever):
        """FAISS should rank GitHub tools highest for a repo search query."""
        results = retriever.retrieve(
            "Search GitHub for Python machine learning repositories sorted by stars",
            candidate_server_ids={"server-github"},
            k=5,
        )

        assert len(results) > 0
        tool_names = [r.tool_name for r in results]
        assert "search_repositories" in tool_names

    def test_faiss_retrieves_file_read(self, retriever):
        """FAISS should find get_file_contents for README retrieval."""
        results = retriever.retrieve(
            "Retrieve the README file contents from the top repository",
            candidate_server_ids={"server-github"},
            k=5,
        )

        assert len(results) > 0
        tool_names = [r.tool_name for r in results]
        assert "get_file_contents" in tool_names

    def test_full_pipeline_github(self, retriever):
        """Full pipeline: GitHub agent should produce valid probe plan."""
        scenario = _get_query_by_id("github_repo_search")
        task_dag = scenario["expected_mock_responses"]["task_analysis"]
        n_nodes = len(task_dag["nodes"])

        mock_llm = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return task_dag
            elif call_count <= 1 + n_nodes:
                # Alignment call (one per DAG node)
                return _make_alignment_response(
                    "S1", "search_repositories", "server-github", "direct", 0.95
                )
            else:
                return _make_probe_response(
                    "P1", "S1", "search_repositories",
                    {"query": "language:python topic:machine-learning stars:>1000", "perPage": 5},
                    difficulty=0.3,
                )

        mock_llm.complete_json.side_effect = side_effect

        pipeline = ProbePipeline(
            llm=mock_llm,
            retriever=retriever,
            config=PipelineConfig(probe_budget=2, total_timeout=30),
        )

        candidates = [
            CandidateAgent(**c) for c in scenario["candidates"]
        ]
        retrieval_result = RetrievalResult(query=scenario["query"], candidates=candidates[:1])

        dag, agent_results = pipeline.run_stages_1_to_4(retrieval_result)

        assert dag is not None
        assert dag.intent == "data_retrieval"
        assert len(dag.nodes) == 3

        result = agent_results[0]
        assert result.error_code is None
        assert result.alignment is not None
        assert result.alignment.coverage_score > 0
        assert result.validated_plan is not None
        assert len(result.validated_plan.probes) >= 1

    def test_scoring_github_pass(self, retriever):
        """Scoring: GitHub agent with successful execution should score high."""
        scenario = _get_query_by_id("github_repo_search")
        task_dag = scenario["expected_mock_responses"]["task_analysis"]
        n_nodes = len(task_dag["nodes"])

        mock_llm = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return task_dag
            elif call_count <= 1 + n_nodes:
                # Alignment call (one per DAG node)
                return _make_alignment_response(
                    "S1", "search_repositories", "server-github", "direct", 0.95
                )
            else:
                return _make_probe_response(
                    "P1", "S1", "search_repositories",
                    {"query": "language:python topic:machine-learning stars:>1000"},
                    difficulty=0.3,
                )

        mock_llm.complete_json.side_effect = side_effect

        pipeline = ProbePipeline(llm=mock_llm, retriever=retriever)
        agent = CandidateAgent(**scenario["candidates"][0])
        retrieval_result = RetrievalResult(query=scenario["query"], candidates=[agent])

        dag, agent_results = pipeline.run_stages_1_to_4(retrieval_result)
        result = agent_results[0]

        # Simulate successful execution
        exec_results = []
        if result.validated_plan:
            for probe in result.validated_plan.probes:
                exec_results.append(_make_execution_result(
                    agent.agent_id, probe.probe_id,
                    {"total_count": 150, "items": [{"full_name": "scikit-learn/scikit-learn"}]},
                ))

        ranked = pipeline.score_agent_results(result, exec_results)

        assert ranked.agent_id == "server-github"
        assert 0.0 <= ranked.theta <= 1.0
        assert ranked.confidence > 0
        assert ranked.testability_tier in ("FULLY_PROBED", "PARTIALLY_PROBED")
        assert "PASS" in ranked.probe_summary


# ===========================================================================
# Test 2: Web Search + Save — multi-agent ranking
# ===========================================================================


class TestWebSearchSave:
    """Query: Search web for AI news and save summary to file."""

    def test_faiss_ranks_brave_for_search(self, retriever):
        """Brave Search should be top result for web search queries."""
        results = retriever.retrieve(
            "Search the web for recent news articles about AI regulation",
            candidate_server_ids={"server-brave-search"},
            k=5,
        )

        assert len(results) > 0
        assert results[0].tool_name == "brave_web_search"

    def test_faiss_ranks_filesystem_for_write(self, retriever):
        """Filesystem should match for file writing subtasks."""
        results = retriever.retrieve(
            "Write the markdown summary to a file on disk",
            candidate_server_ids={"server-filesystem"},
            k=5,
        )

        assert len(results) > 0
        tool_names = [r.tool_name for r in results]
        assert "write_file" in tool_names

    def test_brave_agent_pipeline(self, retriever):
        """Brave search agent should produce valid probes for web search."""
        scenario = _get_query_by_id("web_search_save")
        task_dag = scenario["expected_mock_responses"]["task_analysis"]
        n_nodes = len(task_dag["nodes"])

        mock_llm = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return task_dag
            elif call_count <= 1 + n_nodes:
                # Alignment call (one per DAG node)
                return _make_alignment_response(
                    "S1", "brave_web_search", "server-brave-search", "direct", 0.92
                )
            else:
                return _make_probe_response(
                    "P1", "S1", "brave_web_search",
                    {"query": "EU AI regulation 2025 latest news", "count": 10},
                    difficulty=0.3,
                )

        mock_llm.complete_json.side_effect = side_effect

        pipeline = ProbePipeline(llm=mock_llm, retriever=retriever)
        agent = CandidateAgent(**scenario["candidates"][0])
        retrieval_result = RetrievalResult(query=scenario["query"], candidates=[agent])

        dag, agent_results = pipeline.run_stages_1_to_4(retrieval_result)

        assert dag is not None
        result = agent_results[0]
        assert result.error_code is None
        assert result.validated_plan is not None

    def test_multi_agent_ranking(self, retriever):
        """Brave search should rank higher than fetch for web search."""
        scenario = _get_query_by_id("web_search_save")
        task_dag = scenario["expected_mock_responses"]["task_analysis"]

        # Run pipeline for brave-search (success) vs fetch (partial match)
        n_nodes = len(task_dag["nodes"])
        mock_llm_brave = MagicMock()
        brave_call = 0

        def brave_side_effect(*args, **kwargs):
            nonlocal brave_call
            brave_call += 1
            if brave_call == 1:
                return task_dag
            elif brave_call <= 1 + n_nodes:
                # Alignment call (one per DAG node)
                return _make_alignment_response(
                    "S1", "brave_web_search", "server-brave-search", "direct", 0.92
                )
            else:
                return _make_probe_response(
                    "P1", "S1", "brave_web_search",
                    {"query": "EU AI regulation latest"}, difficulty=0.3,
                )

        mock_llm_brave.complete_json.side_effect = brave_side_effect

        # Brave agent: runs well
        pipeline_brave = ProbePipeline(llm=mock_llm_brave, retriever=retriever)
        agent_brave = CandidateAgent(**scenario["candidates"][0])
        res_brave = RetrievalResult(query=scenario["query"], candidates=[agent_brave])
        dag_brave, results_brave = pipeline_brave.run_stages_1_to_4(res_brave)
        result_brave = results_brave[0]

        exec_brave = []
        if result_brave.validated_plan:
            for p in result_brave.validated_plan.probes:
                exec_brave.append(_make_execution_result(
                    agent_brave.agent_id, p.probe_id,
                    {"results": "AI Act passed..."}, success=True,
                ))
        ranked_brave = pipeline_brave.score_agent_results(result_brave, exec_brave)

        # Fetch agent: partial match, fails execution
        mock_llm_fetch = MagicMock()
        fetch_call = 0

        def fetch_side_effect(*args, **kwargs):
            nonlocal fetch_call
            fetch_call += 1
            if fetch_call == 1:
                return task_dag
            elif fetch_call <= 1 + n_nodes:
                # Alignment call (one per DAG node)
                return _make_alignment_response(
                    "S1", "fetch", "server-fetch", "partial", 0.55
                )
            else:
                return _make_probe_response(
                    "P1", "S1", "fetch",
                    {"url": "https://example.com/ai-news"}, difficulty=0.3,
                )

        mock_llm_fetch.complete_json.side_effect = fetch_side_effect

        pipeline_fetch = ProbePipeline(llm=mock_llm_fetch, retriever=retriever)
        agent_fetch = CandidateAgent(**scenario["candidates"][2])
        res_fetch = RetrievalResult(query=scenario["query"], candidates=[agent_fetch])
        dag_fetch, results_fetch = pipeline_fetch.run_stages_1_to_4(res_fetch)
        result_fetch = results_fetch[0]

        exec_fetch = []
        if result_fetch.validated_plan:
            for p in result_fetch.validated_plan.probes:
                exec_fetch.append(_make_execution_result(
                    agent_fetch.agent_id, p.probe_id,
                    None, success=False, latency_ms=5000,
                ))
        ranked_fetch = pipeline_fetch.score_agent_results(result_fetch, exec_fetch)

        # Brave (direct match + pass) should score higher than fetch (partial + fail)
        assert ranked_brave.theta > ranked_fetch.theta


# ===========================================================================
# Test 3: Database Introspection — single-agent, progressive difficulty
# ===========================================================================


class TestDatabaseIntrospection:
    """Query: List tables, show users schema, count active users."""

    def test_faiss_finds_sqlite_tools(self, retriever):
        """FAISS should retrieve all relevant SQLite tools."""
        results = retriever.retrieve(
            "List all available tables in the SQLite database",
            candidate_server_ids={"server-sqlite"},
            k=5,
        )
        tool_names = [r.tool_name for r in results]
        assert "list_tables" in tool_names

    def test_faiss_finds_describe(self, retriever):
        """FAISS should retrieve describe_table for schema inspection."""
        results = retriever.retrieve(
            "Retrieve the schema definition of the users table",
            candidate_server_ids={"server-sqlite"},
            k=5,
        )
        tool_names = [r.tool_name for r in results]
        assert "describe_table" in tool_names

    def test_faiss_finds_read_query(self, retriever):
        """FAISS should retrieve read_query for SQL execution."""
        results = retriever.retrieve(
            "Execute a SQL query to count active users in the users table",
            candidate_server_ids={"server-sqlite"},
            k=5,
        )
        tool_names = [r.tool_name for r in results]
        assert "read_query" in tool_names

    def test_full_pipeline_sqlite(self, retriever):
        """Full pipeline: SQLite should produce probes for the hard subtask."""
        scenario = _get_query_by_id("database_introspection")
        task_dag = scenario["expected_mock_responses"]["task_analysis"]
        n_nodes = len(task_dag["nodes"])

        mock_llm = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return task_dag
            elif call_count <= 1 + n_nodes:
                # Alignment calls (one per DAG node: S1, S2, S3)
                # Node index for this call
                node_idx = call_count - 2  # 0, 1, 2
                if node_idx == 1:
                    # S2: describe_table
                    return _make_alignment_response(
                        "S2", "describe_table", "server-sqlite", "direct", 0.9
                    )
                elif node_idx == 2:
                    # S3: read_query
                    return _make_alignment_response(
                        "S3", "read_query", "server-sqlite", "direct", 0.88
                    )
                else:
                    # S1: list_tables (non-discriminative)
                    return _make_alignment_response(
                        "S1", "list_tables", "server-sqlite", "direct", 0.85
                    )
            else:
                # Probe generation calls (one per selected subtask)
                probe_idx = call_count - 1 - n_nodes
                if probe_idx == 1:
                    return _make_probe_response(
                        "P1", "S3", "read_query",
                        {"query": "SELECT COUNT(*) as active_count FROM users WHERE status = 'active'"},
                        difficulty=0.4,
                        discrimination=1.5,
                    )
                else:
                    return _make_probe_response(
                        "P2", "S2", "describe_table",
                        {"table_name": "users"},
                        difficulty=0.2,
                    )

        mock_llm.complete_json.side_effect = side_effect

        pipeline = ProbePipeline(
            llm=mock_llm, retriever=retriever,
            config=PipelineConfig(probe_budget=2),
        )
        agent = CandidateAgent(**scenario["candidates"][0])
        retrieval_result = RetrievalResult(query=scenario["query"], candidates=[agent])

        dag, agent_results = pipeline.run_stages_1_to_4(retrieval_result)

        assert dag is not None
        assert dag.domain == "data_engineering"
        result = agent_results[0]
        assert result.error_code is None
        assert result.validated_plan is not None
        # Should have selected the discriminative subtasks (S2, S3)
        probed_subtasks = {p.targets_subtask for p in result.validated_plan.probes}
        assert "S2" in probed_subtasks or "S3" in probed_subtasks

    def test_scoring_sqlite_full(self, retriever):
        """SQLite agent passing all probes should be FULLY_PROBED."""
        scenario = _get_query_by_id("database_introspection")
        task_dag = scenario["expected_mock_responses"]["task_analysis"]
        n_nodes = len(task_dag["nodes"])

        mock_llm = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return task_dag
            elif call_count <= 1 + n_nodes:
                # Alignment call (one per DAG node); subtask_id is
                # overridden by align_tools_for_agent to the correct node ID.
                return _make_alignment_response(
                    "S3", "read_query", "server-sqlite", "direct", 0.92
                )
            else:
                return _make_probe_response(
                    f"P{call_count - 1 - n_nodes}", "S3", "read_query",
                    {"query": "SELECT COUNT(*) FROM users WHERE status = 'active'"},
                    difficulty=0.4,
                )

        mock_llm.complete_json.side_effect = side_effect

        pipeline = ProbePipeline(llm=mock_llm, retriever=retriever)
        agent = CandidateAgent(**scenario["candidates"][0])
        retrieval_result = RetrievalResult(query=scenario["query"], candidates=[agent])

        dag, agent_results = pipeline.run_stages_1_to_4(retrieval_result)
        result = agent_results[0]

        exec_results = []
        if result.validated_plan:
            for probe in result.validated_plan.probes:
                exec_results.append(_make_execution_result(
                    agent.agent_id, probe.probe_id,
                    [{"active_count": 1523}],
                ))

        ranked = pipeline.score_agent_results(result, exec_results)
        assert ranked.theta > 0.5
        assert ranked.testability_tier in ("FULLY_PROBED", "PARTIALLY_PROBED")


# ===========================================================================
# Test 4: Web Scraping — multi-tool, high complexity
# ===========================================================================


class TestWebScraping:
    """Query: Navigate, screenshot, extract, save."""

    def test_faiss_puppeteer_navigation(self, retriever):
        """FAISS should rank puppeteer_navigate for navigation tasks."""
        results = retriever.retrieve(
            "Navigate the browser to the target product page URL",
            candidate_server_ids={"server-puppeteer"},
            k=5,
        )
        tool_names = [r.tool_name for r in results]
        assert "puppeteer_navigate" in tool_names

    def test_faiss_puppeteer_evaluate(self, retriever):
        """FAISS should rank puppeteer_evaluate for DOM extraction."""
        results = retriever.retrieve(
            "Execute JavaScript to extract the product title and price from the page DOM",
            candidate_server_ids={"server-puppeteer"},
            k=5,
        )
        tool_names = [r.tool_name for r in results]
        assert "puppeteer_evaluate" in tool_names

    def test_puppeteer_over_fetch(self, retriever):
        """Puppeteer should outrank fetch for scraping tasks."""
        scenario = _get_query_by_id("web_scraping")
        task_dag = scenario["expected_mock_responses"]["task_analysis"]
        n_nodes = len(task_dag["nodes"])

        # Puppeteer pipeline — direct match, success
        mock_llm_pup = MagicMock()
        pup_call = 0

        def pup_side_effect(*args, **kwargs):
            nonlocal pup_call
            pup_call += 1
            if pup_call == 1:
                return task_dag
            elif pup_call <= 1 + n_nodes:
                # Alignment call (one per DAG node)
                return _make_alignment_response(
                    "S3", "puppeteer_evaluate", "server-puppeteer", "direct", 0.9
                )
            else:
                return _make_probe_response(
                    "P1", "S3", "puppeteer_evaluate",
                    {"script": "JSON.stringify({title: document.querySelector('h1').textContent, price: document.querySelector('.price').textContent})"},
                    difficulty=0.6,
                    discrimination=1.8,
                )

        mock_llm_pup.complete_json.side_effect = pup_side_effect

        pipeline_pup = ProbePipeline(llm=mock_llm_pup, retriever=retriever)
        agent_pup = CandidateAgent(**scenario["candidates"][0])
        res_pup = RetrievalResult(query=scenario["query"], candidates=[agent_pup])
        dag_pup, results_pup = pipeline_pup.run_stages_1_to_4(res_pup)
        result_pup = results_pup[0]

        exec_pup = []
        if result_pup.validated_plan:
            for p in result_pup.validated_plan.probes:
                exec_pup.append(_make_execution_result(
                    agent_pup.agent_id, p.probe_id,
                    {"title": "Widget Pro", "price": "$29.99"}, success=True,
                ))
        ranked_pup = pipeline_pup.score_agent_results(result_pup, exec_pup)

        # Fetch pipeline — partial match, fetch can't extract DOM
        mock_llm_fetch = MagicMock()
        fetch_call = 0

        def fetch_side_effect(*args, **kwargs):
            nonlocal fetch_call
            fetch_call += 1
            if fetch_call == 1:
                return task_dag
            elif fetch_call <= 1 + n_nodes:
                # Alignment call (one per DAG node)
                return _make_alignment_response(
                    "S3", "fetch", "server-fetch", "partial", 0.4
                )
            else:
                return _make_probe_response(
                    "P1", "S3", "fetch",
                    {"url": "https://example.com/product"},
                    difficulty=0.6,
                )

        mock_llm_fetch.complete_json.side_effect = fetch_side_effect

        pipeline_fetch = ProbePipeline(llm=mock_llm_fetch, retriever=retriever)
        agent_fetch = CandidateAgent(**scenario["candidates"][1])
        res_fetch = RetrievalResult(query=scenario["query"], candidates=[agent_fetch])
        dag_fetch, results_fetch = pipeline_fetch.run_stages_1_to_4(res_fetch)
        result_fetch = results_fetch[0]

        exec_fetch = []
        if result_fetch.validated_plan:
            for p in result_fetch.validated_plan.probes:
                exec_fetch.append(_make_execution_result(
                    agent_fetch.agent_id, p.probe_id,
                    "<html>...</html>", success=False,
                ))
        ranked_fetch = pipeline_fetch.score_agent_results(result_fetch, exec_fetch)

        assert ranked_pup.theta > ranked_fetch.theta


# ===========================================================================
# Test 5: Slack Notification — discrimination / negative match
# ===========================================================================


class TestSlackNotification:
    """Query: Send deployment notification to Slack channel."""

    def test_faiss_slack_over_github(self, retriever):
        """FAISS should rank Slack tools higher than GitHub for messaging."""
        slack_results = retriever.retrieve(
            "Post the deployment notification message to the #engineering channel",
            candidate_server_ids={"server-slack"},
            k=5,
        )
        github_results = retriever.retrieve(
            "Post the deployment notification message to the #engineering channel",
            candidate_server_ids={"server-github"},
            k=5,
        )

        assert len(slack_results) > 0
        slack_tool = slack_results[0]
        assert slack_tool.tool_name == "slack_post_message"

        # GitHub may return something, but score should be lower
        if github_results:
            assert slack_tool.similarity_score > github_results[0].similarity_score

    def test_filesystem_no_match_for_messaging(self, retriever):
        """Filesystem should have poor matches for messaging tasks."""
        results = retriever.retrieve(
            "Post the deployment notification message to the #engineering channel",
            candidate_server_ids={"server-filesystem"},
            k=5,
        )
        # Even if results are returned, similarity should be low
        if results:
            assert results[0].similarity_score < 0.5

    def test_slack_beats_github_and_filesystem(self, retriever):
        """Slack should rank highest for notification tasks, GitHub and filesystem low."""
        scenario = _get_query_by_id("slack_notification")
        task_dag = scenario["expected_mock_responses"]["task_analysis"]
        n_nodes = len(task_dag["nodes"])

        def run_agent(candidate_dict, alignment_response, probe_args, success):
            mock_llm = MagicMock()
            call_count_inner = 0

            def side_effect(*args, **kwargs):
                nonlocal call_count_inner
                call_count_inner += 1
                if call_count_inner == 1:
                    return task_dag
                elif call_count_inner <= 1 + n_nodes:
                    # Alignment call (one per DAG node)
                    return alignment_response
                else:
                    # Probe generation — only reached if alignments exist
                    tool_name = (
                        alignment_response["alignments"][0]["tool_name"]
                        if alignment_response.get("alignments")
                        else "unknown"
                    )
                    return _make_probe_response(
                        "P1", "S3", tool_name,
                        probe_args, difficulty=0.25,
                    )

            mock_llm.complete_json.side_effect = side_effect

            pipeline = ProbePipeline(llm=mock_llm, retriever=retriever)
            agent = CandidateAgent(**candidate_dict)
            res = RetrievalResult(query=scenario["query"], candidates=[agent])
            dag, results = pipeline.run_stages_1_to_4(res)
            result = results[0]

            exec_results = []
            if result.validated_plan:
                for p in result.validated_plan.probes:
                    exec_results.append(_make_execution_result(
                        agent.agent_id, p.probe_id,
                        {"ok": True} if success else None,
                        success=success,
                    ))

            return pipeline.score_agent_results(result, exec_results)

        # Slack: direct match, succeeds
        ranked_slack = run_agent(
            scenario["candidates"][0],
            _make_alignment_response("S3", "slack_post_message", "server-slack", "direct", 0.95),
            {"channel_id": "C12345", "text": "Deployment complete"},
            success=True,
        )

        # GitHub: poor match, fails
        ranked_github = run_agent(
            scenario["candidates"][1],
            _make_alignment_response("S3", "create_issue", "server-github", "inferred", 0.3),
            {"owner": "org", "repo": "infra", "title": "Deployment notification"},
            success=False,
        )

        # Filesystem: no meaningful match, fails
        ranked_fs = run_agent(
            scenario["candidates"][2],
            {"alignments": []},
            {},
            success=False,
        )

        # Slack should win
        assert ranked_slack.theta > ranked_github.theta
        assert ranked_slack.theta > ranked_fs.theta
        assert ranked_slack.confidence >= ranked_fs.confidence


# ===========================================================================
# Test 6: Cross-cutting — FAISS index covers all 7 servers
# ===========================================================================


class TestCrossCutting:
    """Verify the FAISS index is coherent across all 7 real servers."""

    def test_total_tool_count(self, retriever):
        """Index should contain all tools from all 7 servers."""
        servers = _load_real_servers()
        expected_count = sum(len(s["tools"]) for s in servers)
        assert len(retriever.tools) == expected_count

    def test_all_servers_represented(self, retriever):
        """Every server should have at least one tool in the index."""
        server_ids = {t.server_id for t in retriever.tools}
        expected_servers = {
            "server-github", "server-filesystem", "server-brave-search",
            "server-sqlite", "server-puppeteer", "server-fetch", "server-slack",
        }
        assert server_ids == expected_servers

    def test_unfiltered_retrieval_spans_servers(self, retriever):
        """A broad query should return tools from multiple servers."""
        results = retriever.retrieve(
            "Search for information online",
            k=10,
        )
        servers_found = {r.server_id for r in results}
        # Should match at least 2 different servers
        assert len(servers_found) >= 2

    def test_server_filter_isolation(self, retriever):
        """Filtering by server_id should return only that server's tools."""
        for server_id in ["server-github", "server-sqlite", "server-slack"]:
            results = retriever.retrieve(
                "do something",
                candidate_server_ids={server_id},
                k=10,
            )
            for r in results:
                assert r.server_id == server_id
