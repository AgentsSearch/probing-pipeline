"""Interaction confidence — rule-based checks on probe execution results.

Run BEFORE the LLM judge. Checks:
  1. Tool discovery succeeded (list_tools returned non-empty)
  2. Tool call arguments were schema-valid
  3. Response received (no HTTP error, no timeout)
  4. Response contains non-trivial content

If confidence < 0.5: discard result, classify as PARTIALLY_PROBED.
If confidence >= 0.5: proceed to LLM judge evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.models.integration import ProbeExecutionResult

logger = logging.getLogger(__name__)

# Weight for each confidence check
_WEIGHTS = {
    "tool_discovery": 0.25,
    "schema_valid": 0.25,
    "response_received": 0.30,
    "nontrivial_content": 0.20,
}

_MIN_CONFIDENCE = 0.5


@dataclass
class ConfidenceAssessment:
    """Result of interaction confidence assessment."""

    score: float
    sufficient: bool
    checks: dict[str, bool]
    reason: str | None = None


def _check_tool_discovery(result: ProbeExecutionResult) -> bool:
    """Check that tool discovery succeeded during execution."""
    if not result.trajectory:
        return False
    # Look for a successful list_tools or similar discovery step
    for step in result.trajectory:
        if step.action in ("list_tools", "tool_discovery") and step.result:
            return True
    # If trajectory exists and no explicit discovery, assume it worked
    # (the tool was called, which implies discovery succeeded)
    return result.success or any(s.tool_name is not None for s in result.trajectory)


def _check_schema_valid(result: ProbeExecutionResult) -> bool:
    """Check that tool call arguments were schema-valid."""
    for step in result.trajectory:
        if step.tool_name and step.error and "schema" in step.error.lower():
            return False
        if step.tool_name and step.error and "invalid" in step.error.lower():
            return False
    return True


def _check_response_received(result: ProbeExecutionResult) -> bool:
    """Check that a response was received (no HTTP error, no timeout)."""
    if result.error_info:
        error_lower = result.error_info.lower()
        if any(kw in error_lower for kw in ("timeout", "http error", "connection")):
            return False
    return result.success or result.output is not None


def _check_nontrivial_content(result: ProbeExecutionResult) -> bool:
    """Check that the response contains meaningful content."""
    if result.output is None:
        return False
    if isinstance(result.output, str):
        return len(result.output.strip()) > 0
    if isinstance(result.output, dict):
        return len(result.output) > 0
    if isinstance(result.output, list):
        return len(result.output) > 0
    # For other types, assume non-trivial if not None
    return True


def assess_confidence(result: ProbeExecutionResult) -> ConfidenceAssessment:
    """Assess interaction confidence for a probe execution result.

    Args:
        result: The execution result from Stream C sandbox.

    Returns:
        ConfidenceAssessment with score, sufficiency flag, and per-check results.
    """
    checks = {
        "tool_discovery": _check_tool_discovery(result),
        "schema_valid": _check_schema_valid(result),
        "response_received": _check_response_received(result),
        "nontrivial_content": _check_nontrivial_content(result),
    }

    score = sum(
        _WEIGHTS[check] for check, passed in checks.items() if passed
    )

    sufficient = score >= _MIN_CONFIDENCE

    reason = None
    if not sufficient:
        failed = [name for name, passed in checks.items() if not passed]
        reason = f"Low confidence ({score:.2f}): failed checks: {', '.join(failed)}"

    assessment = ConfidenceAssessment(
        score=score,
        sufficient=sufficient,
        checks=checks,
        reason=reason,
    )

    logger.info(
        "Confidence for %s/%s: %.2f (%s) checks=%s",
        result.agent_id,
        result.probe_id,
        score,
        "PASS" if sufficient else "FAIL",
        checks,
    )

    return assessment
