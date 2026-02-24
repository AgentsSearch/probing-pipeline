"""Back-instruct template generator — creates probe templates from tool schemas.

Adapted from TaskBench's back-instruct method. Generates 2-3 probe templates
per tool at varying difficulty levels.
"""

from __future__ import annotations

import logging
from typing import Any

from src.llm.client import LLMClient
from src.models.probe import ProbeTemplate, RubricDimension

logger = logging.getLogger(__name__)

_SYSTEM = "You are a probe template generator for an agent evaluation system. Return only valid JSON."

_GENERATE_PROMPT = """Given an MCP tool schema, generate a probe template at the specified difficulty level.

Tool name: {tool_name}
Server ID: {server_id}
Tool description: {tool_description}
Parameter schema: {parameter_schema}
Target difficulty: {difficulty} (0=trivial, 1=very hard)

Return a JSON object:
{{
  "arg_template": {{ "<param>": "{{placeholder}}" }},
  "expected_behaviour": "<what a correct execution looks like>",
  "discrimination": <float 0.5-2.0>,
  "rubric_template": [
    {{
      "name": "<dimension>",
      "weight": <float>,
      "criteria": "<what to evaluate>",
      "pass_threshold": "<what constitutes passing>"
    }}
  ]
}}

Rules:
1. Use {{placeholder}} syntax for values that vary per query (e.g. {{city_name}}).
2. Rubric must have 2-4 dimensions with weights summing to 1.0.
3. Return ONLY the JSON object."""


def generate_template(
    tool_name: str,
    server_id: str,
    tool_description: str,
    parameter_schema: dict[str, Any],
    difficulty: float,
    llm: LLMClient,
) -> ProbeTemplate:
    """Generate a single probe template for a tool at a given difficulty.

    Args:
        tool_name: MCP tool name.
        server_id: Server ID the tool belongs to.
        tool_description: Natural language description of the tool.
        parameter_schema: JSON schema of the tool's parameters.
        difficulty: Target difficulty level (0-1).
        llm: Initialised LLMClient.

    Returns:
        A ProbeTemplate ready to be added to the library.
    """
    import json

    prompt = (
        _GENERATE_PROMPT
        .replace("{tool_name}", tool_name)
        .replace("{server_id}", server_id)
        .replace("{tool_description}", tool_description)
        .replace("{parameter_schema}", json.dumps(parameter_schema, indent=2))
        .replace("{difficulty}", str(difficulty))
    )

    data = llm.complete_json(prompt, system=_SYSTEM)

    rubric = [
        RubricDimension(
            name=r["name"],
            weight=float(r["weight"]),
            criteria=r["criteria"],
            pass_threshold=r["pass_threshold"],
        )
        for r in data.get("rubric_template", [])
    ]

    template_id = f"{server_id}__{tool_name}__d{int(difficulty * 10)}"

    return ProbeTemplate(
        template_id=template_id,
        server_id=server_id,
        tool_name=tool_name,
        difficulty=difficulty,
        discrimination=float(data.get("discrimination", 1.0)),
        arg_template=data.get("arg_template", {}),
        expected_behaviour=data.get("expected_behaviour", ""),
        rubric_template=rubric,
        validated=False,
    )


def generate_templates_for_tool(
    tool_name: str,
    server_id: str,
    tool_description: str,
    parameter_schema: dict[str, Any],
    llm: LLMClient,
    difficulty_levels: list[float] | None = None,
) -> list[ProbeTemplate]:
    """Generate multiple probe templates at varying difficulty levels.

    Args:
        tool_name: MCP tool name.
        server_id: Server ID.
        tool_description: Tool description.
        parameter_schema: Tool parameter schema.
        llm: Initialised LLMClient.
        difficulty_levels: List of difficulty values. Defaults to [0.2, 0.5, 0.8].

    Returns:
        List of generated ProbeTemplates.
    """
    if difficulty_levels is None:
        difficulty_levels = [0.2, 0.5, 0.8]

    templates: list[ProbeTemplate] = []
    for diff in difficulty_levels:
        try:
            tmpl = generate_template(
                tool_name, server_id, tool_description, parameter_schema, diff, llm
            )
            templates.append(tmpl)
        except Exception as e:
            logger.error(
                "Failed to generate template for %s at difficulty %.1f: %s",
                tool_name, diff, e,
            )

    return templates
