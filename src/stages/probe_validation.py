"""Stage 4: Probe Validation — rule-based checks on generated probes.

Input:  ProbePlan
Output: Validated ProbePlan (or regenerated probes)
Cost:   0 LLM calls (rule-based for MVP)

Checks:
  1. Schema validity: probe arguments type-check against MCP tool parameter schema
  2. Rubric specificity: at least 2 dimensions with non-empty pass thresholds
  3. Difficulty calibration: probe difficulty within tolerance of tool complexity
  4. Timeout feasibility: timeout >= minimum threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.models.probe import Probe, ProbePlan

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single probe."""

    probe_id: str
    valid: bool
    errors: list[str]


def _validate_rubric(probe: Probe, min_dimensions: int = 2) -> list[str]:
    """Check rubric has enough dimensions with non-empty thresholds."""
    errors: list[str] = []

    if len(probe.rubric) < min_dimensions:
        errors.append(
            f"Rubric has {len(probe.rubric)} dimensions, minimum is {min_dimensions}"
        )

    for dim in probe.rubric:
        if not dim.pass_threshold or not dim.pass_threshold.strip():
            errors.append(f"Rubric dimension '{dim.name}' has empty pass_threshold")

    # Check weights sum approximately to 1.0
    total_weight = sum(d.weight for d in probe.rubric)
    if probe.rubric and abs(total_weight - 1.0) > 0.05:
        errors.append(f"Rubric weights sum to {total_weight:.2f}, expected ~1.0")

    return errors


def _validate_difficulty(
    probe: Probe,
    tool_complexity: float | None = None,
    tolerance: float = 0.3,
) -> list[str]:
    """Check probe difficulty is calibrated against tool complexity."""
    errors: list[str] = []

    if not 0.0 <= probe.estimated_difficulty <= 1.0:
        errors.append(
            f"Difficulty {probe.estimated_difficulty} outside valid range [0, 1]"
        )

    if tool_complexity is not None:
        diff = abs(probe.estimated_difficulty - tool_complexity)
        if diff > tolerance:
            errors.append(
                f"Difficulty {probe.estimated_difficulty:.2f} deviates "
                f"{diff:.2f} from tool complexity {tool_complexity:.2f} "
                f"(tolerance={tolerance})"
            )

    return errors


def _validate_timeout(probe: Probe, min_timeout: int = 5) -> list[str]:
    """Check timeout is reasonable."""
    errors: list[str] = []
    if probe.timeout_seconds < min_timeout:
        errors.append(
            f"Timeout {probe.timeout_seconds}s below minimum {min_timeout}s"
        )
    return errors


def _validate_schema(
    probe: Probe,
    tool_schema: dict[str, Any] | None = None,
) -> list[str]:
    """Check probe arguments against the tool's parameter schema.

    Basic type checking when schema is provided. If no schema is available,
    only checks that arguments is non-empty.
    """
    errors: list[str] = []

    if not probe.arguments:
        # Empty arguments are valid if the tool has no required params
        if tool_schema is not None and not tool_schema.get("required"):
            return errors
        errors.append("Probe has no arguments")
        return errors

    if tool_schema is None:
        return errors

    required = tool_schema.get("required", [])
    properties = tool_schema.get("properties", {})

    for req_param in required:
        if req_param not in probe.arguments:
            errors.append(f"Missing required parameter: {req_param}")

    for param, value in probe.arguments.items():
        if param in properties:
            expected_type = properties[param].get("type")
            if expected_type and not _type_matches(value, expected_type):
                errors.append(
                    f"Parameter '{param}': expected type '{expected_type}', "
                    f"got {type(value).__name__}"
                )

    return errors


def _type_matches(value: Any, json_type: str) -> bool:
    """Check if a Python value matches a JSON schema type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    expected = type_map.get(json_type)
    if expected is None:
        return True
    return isinstance(value, expected)


def validate_probe(
    probe: Probe,
    tool_schema: dict[str, Any] | None = None,
    tool_complexity: float | None = None,
    difficulty_tolerance: float = 0.3,
    min_rubric_dimensions: int = 2,
) -> ValidationResult:
    """Validate a single probe against all rule-based checks.

    Args:
        probe: The probe to validate.
        tool_schema: JSON schema of the tool's parameters (optional).
        tool_complexity: Estimated complexity of the tool (optional).
        difficulty_tolerance: Max allowed deviation from tool complexity.
        min_rubric_dimensions: Minimum rubric dimensions required.

    Returns:
        ValidationResult with errors list.
    """
    errors: list[str] = []
    errors.extend(_validate_schema(probe, tool_schema))
    errors.extend(_validate_rubric(probe, min_rubric_dimensions))
    errors.extend(_validate_difficulty(probe, tool_complexity, difficulty_tolerance))
    errors.extend(_validate_timeout(probe))

    result = ValidationResult(
        probe_id=probe.probe_id,
        valid=len(errors) == 0,
        errors=errors,
    )

    if not result.valid:
        logger.warning("Probe %s failed validation: %s", probe.probe_id, errors)

    return result


def validate_plan(
    plan: ProbePlan,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
    tool_complexities: dict[str, float] | None = None,
    difficulty_tolerance: float = 0.3,
    min_rubric_dimensions: int = 2,
) -> tuple[ProbePlan, list[ValidationResult]]:
    """Validate all probes in a plan, removing invalid ones.

    Args:
        plan: The ProbePlan to validate.
        tool_schemas: Map of tool_name -> parameter schema.
        tool_complexities: Map of tool_name -> complexity estimate.
        difficulty_tolerance: Max difficulty deviation allowed.
        min_rubric_dimensions: Minimum rubric dimensions.

    Returns:
        Tuple of (validated ProbePlan with only valid probes, all ValidationResults).
    """
    tool_schemas = tool_schemas or {}
    tool_complexities = tool_complexities or {}

    results: list[ValidationResult] = []
    valid_probes: list[Probe] = []

    for probe in plan.probes:
        result = validate_probe(
            probe,
            tool_schema=tool_schemas.get(probe.tool),
            tool_complexity=tool_complexities.get(probe.tool),
            difficulty_tolerance=difficulty_tolerance,
            min_rubric_dimensions=min_rubric_dimensions,
        )
        results.append(result)
        if result.valid:
            valid_probes.append(probe)

    validated_plan = ProbePlan(
        query=plan.query,
        agent_id=plan.agent_id,
        strategy=plan.strategy,
        probes=valid_probes,
        total_budget_seconds=plan.total_budget_seconds,
    )

    logger.info(
        "Validation for %s: %d/%d probes passed",
        plan.agent_id,
        len(valid_probes),
        len(plan.probes),
    )

    return validated_plan, results
