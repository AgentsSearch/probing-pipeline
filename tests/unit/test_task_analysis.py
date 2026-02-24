"""Unit tests for Stage 1: Task Analysis."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.task import SubtaskNode, TaskDAG
from src.stages.task_analysis import _merge_excess_nodes, _parse_task_dag, analyse_task


class TestParseTaskDAG:
    def _mock_data(self, **overrides):
        data = {
            "query": "Find weather in London",
            "intent": "data_retrieval",
            "domain": "general",
            "nodes": [
                {
                    "id": "S1",
                    "description": "Geocode location",
                    "capability": "geocoding",
                    "difficulty": 0.2,
                    "is_discriminative": False,
                    "depends_on": [],
                },
                {
                    "id": "S2",
                    "description": "Fetch weather data",
                    "capability": "api_call",
                    "difficulty": 0.5,
                    "is_discriminative": True,
                    "depends_on": ["S1"],
                },
            ],
            "edges": [["S1", "S2"]],
            "critical_path": ["S1", "S2"],
            "estimated_difficulty": 0.4,
            "evaluation_dimensions": ["correctness"],
        }
        data.update(overrides)
        return data

    def test_valid_parse(self):
        data = self._mock_data()
        dag = _parse_task_dag("Find weather in London", data)
        assert isinstance(dag, TaskDAG)
        assert len(dag.nodes) == 2
        assert dag.nodes[1].is_discriminative is True
        assert dag.critical_path == ["S1", "S2"]

    def test_missing_query_uses_fallback(self):
        data = self._mock_data()
        del data["query"]
        dag = _parse_task_dag("my fallback query", data)
        assert dag.query == "my fallback query"

    def test_missing_optional_fields(self):
        data = self._mock_data()
        del data["evaluation_dimensions"]
        dag = _parse_task_dag("test", data)
        assert dag.evaluation_dimensions == []

    def test_excess_nodes_merged(self):
        nodes = [
            {
                "id": f"S{i}",
                "description": f"task {i}",
                "capability": "general",
                "difficulty": i * 0.1,
                "is_discriminative": False,
                "depends_on": [],
            }
            for i in range(1, 9)
        ]
        data = self._mock_data(nodes=nodes)
        dag = _parse_task_dag("test", data)
        assert len(dag.nodes) <= 6


class TestMergeExcessNodes:
    def test_merges_to_target(self):
        nodes = [
            {"id": f"S{i}", "description": f"task {i}", "capability": "general",
             "difficulty": i * 0.1, "is_discriminative": False, "depends_on": []}
            for i in range(1, 9)
        ]
        merged = _merge_excess_nodes(nodes, 6)
        assert len(merged) == 6

    def test_preserves_discriminative(self):
        nodes = [
            {"id": "S1", "description": "easy", "capability": "x",
             "difficulty": 0.1, "is_discriminative": False, "depends_on": []},
            {"id": "S2", "description": "hard", "capability": "x",
             "difficulty": 0.9, "is_discriminative": True, "depends_on": []},
            {"id": "S3", "description": "medium", "capability": "x",
             "difficulty": 0.5, "is_discriminative": False, "depends_on": []},
        ]
        merged = _merge_excess_nodes(nodes, 2)
        assert len(merged) == 2
        # The discriminative node should survive
        disc_ids = [n["id"] for n in merged if n.get("is_discriminative")]
        assert len(disc_ids) >= 1

    def test_no_op_when_within_limit(self):
        nodes = [
            {"id": "S1", "description": "t", "capability": "x",
             "difficulty": 0.5, "is_discriminative": False, "depends_on": []}
        ]
        merged = _merge_excess_nodes(nodes, 6)
        assert len(merged) == 1


class TestAnalyseTask:
    def test_calls_llm_and_returns_dag(self):
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {
            "query": "test query",
            "intent": "data_retrieval",
            "domain": "general",
            "nodes": [
                {"id": "S1", "description": "step 1", "capability": "api_call",
                 "difficulty": 0.3, "is_discriminative": False, "depends_on": []},
                {"id": "S2", "description": "step 2", "capability": "transform",
                 "difficulty": 0.6, "is_discriminative": True, "depends_on": ["S1"]},
            ],
            "edges": [["S1", "S2"]],
            "critical_path": ["S1", "S2"],
            "estimated_difficulty": 0.45,
            "evaluation_dimensions": ["correctness", "completeness"],
        }

        dag = analyse_task("test query", mock_llm)
        assert isinstance(dag, TaskDAG)
        assert len(dag.nodes) == 2
        mock_llm.complete_json.assert_called_once()

    def test_raises_on_invalid_llm_output(self):
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = {"nodes": "not a list"}

        with pytest.raises(ValueError, match="Invalid TaskDAG"):
            analyse_task("test", mock_llm)
