"""Unit tests for Stage 4: Probe Validation."""

import pytest

from src.models.probe import Probe, ProbePlan, RubricDimension
from src.stages.probe_validation import (
    ValidationResult,
    validate_probe,
    validate_plan,
    _type_matches,
    _validate_rubric,
    _validate_difficulty,
    _validate_schema,
    _validate_timeout,
)


def _make_rubric(n=2, weight_each=None) -> list[RubricDimension]:
    w = weight_each if weight_each is not None else 1.0 / n
    return [
        RubricDimension(name=f"dim{i}", weight=w, criteria="c", pass_threshold="p")
        for i in range(n)
    ]


def _make_probe(**overrides) -> Probe:
    defaults = dict(
        probe_id="P1", targets_subtask="S1", tool="test_tool",
        arguments={"query": "hello"}, estimated_difficulty=0.5,
        discrimination=1.0, rubric=_make_rubric(2),
        timeout_seconds=15, priority="PRIMARY",
    )
    defaults.update(overrides)
    return Probe(**defaults)


class TestTypeMatches:
    def test_string(self):
        assert _type_matches("hello", "string") is True
        assert _type_matches(123, "string") is False

    def test_integer(self):
        assert _type_matches(42, "integer") is True
        assert _type_matches(3.14, "integer") is False

    def test_number(self):
        assert _type_matches(42, "number") is True
        assert _type_matches(3.14, "number") is True
        assert _type_matches("x", "number") is False

    def test_boolean(self):
        assert _type_matches(True, "boolean") is True
        assert _type_matches(0, "boolean") is False

    def test_array(self):
        assert _type_matches([1, 2], "array") is True
        assert _type_matches({}, "array") is False

    def test_object(self):
        assert _type_matches({"a": 1}, "object") is True
        assert _type_matches([], "object") is False

    def test_unknown_type_passes(self):
        assert _type_matches("anything", "unknown_type") is True


class TestValidateRubric:
    def test_valid_rubric(self):
        probe = _make_probe(rubric=_make_rubric(3, weight_each=1.0 / 3))
        errors = _validate_rubric(probe, min_dimensions=2)
        assert errors == []

    def test_too_few_dimensions(self):
        # Can't use _make_probe because Probe.__post_init__ enforces >= 2
        # Test the function directly with a mock-like object
        class FakeProbe:
            rubric = [RubricDimension(name="x", weight=1.0, criteria="c", pass_threshold="p")]
        errors = _validate_rubric(FakeProbe(), min_dimensions=2)
        assert any("dimensions" in e for e in errors)

    def test_empty_pass_threshold(self):
        rubric = [
            RubricDimension(name="a", weight=0.5, criteria="c", pass_threshold="ok"),
            RubricDimension(name="b", weight=0.5, criteria="c", pass_threshold=""),
        ]
        probe = _make_probe(rubric=rubric)
        errors = _validate_rubric(probe)
        assert any("empty pass_threshold" in e for e in errors)

    def test_weights_dont_sum_to_one(self):
        probe = _make_probe(rubric=_make_rubric(2, weight_each=0.3))
        errors = _validate_rubric(probe)
        assert any("weights sum" in e for e in errors)


class TestValidateDifficulty:
    def test_valid_difficulty(self):
        probe = _make_probe(estimated_difficulty=0.5)
        errors = _validate_difficulty(probe, tool_complexity=0.6, tolerance=0.3)
        assert errors == []

    def test_out_of_tolerance(self):
        probe = _make_probe(estimated_difficulty=0.2)
        errors = _validate_difficulty(probe, tool_complexity=0.8, tolerance=0.3)
        assert any("deviates" in e for e in errors)

    def test_no_tool_complexity(self):
        probe = _make_probe(estimated_difficulty=0.5)
        errors = _validate_difficulty(probe, tool_complexity=None)
        assert errors == []


class TestValidateSchema:
    def test_no_schema_no_errors(self):
        probe = _make_probe(arguments={"x": 1})
        errors = _validate_schema(probe, tool_schema=None)
        assert errors == []

    def test_missing_required_param(self):
        probe = _make_probe(arguments={"x": 1})
        schema = {"required": ["x", "y"], "properties": {}}
        errors = _validate_schema(probe, tool_schema=schema)
        assert any("Missing required parameter: y" in e for e in errors)

    def test_type_mismatch(self):
        probe = _make_probe(arguments={"x": 123})
        schema = {"required": [], "properties": {"x": {"type": "string"}}}
        errors = _validate_schema(probe, tool_schema=schema)
        assert any("expected type" in e for e in errors)

    def test_empty_arguments(self):
        probe = _make_probe(arguments={})
        errors = _validate_schema(probe, tool_schema=None)
        assert any("no arguments" in e for e in errors)


class TestValidateTimeout:
    def test_valid_timeout(self):
        probe = _make_probe(timeout_seconds=15)
        errors = _validate_timeout(probe, min_timeout=5)
        assert errors == []

    def test_too_low_timeout(self):
        probe = _make_probe(timeout_seconds=2)
        errors = _validate_timeout(probe, min_timeout=5)
        assert any("below minimum" in e for e in errors)


class TestValidateProbe:
    def test_fully_valid_probe(self):
        probe = _make_probe()
        result = validate_probe(probe)
        assert result.valid is True
        assert result.errors == []

    def test_multiple_errors(self):
        probe = _make_probe(
            arguments={},
            rubric=_make_rubric(2, weight_each=0.3),
            timeout_seconds=2,
        )
        result = validate_probe(probe, min_rubric_dimensions=2)
        assert result.valid is False
        assert len(result.errors) >= 2


class TestValidatePlan:
    def test_filters_invalid_probes(self):
        good_probe = _make_probe(probe_id="P1")
        bad_probe = _make_probe(probe_id="P2", arguments={}, timeout_seconds=2)
        plan = ProbePlan(
            query="test", agent_id="a1",
            strategy="discriminative_critical_path",
            probes=[good_probe, bad_probe],
            total_budget_seconds=30,
        )
        validated, results = validate_plan(plan)
        assert len(validated.probes) == 1
        assert validated.probes[0].probe_id == "P1"
        assert len(results) == 2

    def test_all_valid(self):
        plan = ProbePlan(
            query="test", agent_id="a1",
            strategy="discriminative_critical_path",
            probes=[_make_probe(probe_id="P1"), _make_probe(probe_id="P2")],
            total_budget_seconds=30,
        )
        validated, results = validate_plan(plan)
        assert len(validated.probes) == 2
        assert all(r.valid for r in results)
