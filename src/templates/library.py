"""Probe template library — stores and looks up pre-computed probe templates.

Templates are matched by tool name and difficulty level, avoiding LLM calls
when a suitable template exists.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.models.probe import ProbeTemplate, RubricDimension

logger = logging.getLogger(__name__)


class TemplateLibrary:
    """In-memory probe template store with persistence.

    Args:
        storage_path: Path to the JSON file for template persistence.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._templates: list[ProbeTemplate] = []
        self._storage_path = Path(storage_path) if storage_path else None
        if self._storage_path and self._storage_path.exists():
            self._load()

    def _load(self) -> None:
        """Load templates from disk."""
        with open(self._storage_path) as f:
            raw = json.load(f)
        for item in raw:
            rubric = [
                RubricDimension(**r) for r in item.pop("rubric_template", [])
            ]
            item.pop("created_at", None)
            self._templates.append(ProbeTemplate(**item, rubric_template=rubric))
        logger.info("Loaded %d templates from %s", len(self._templates), self._storage_path)

    def save(self) -> None:
        """Persist all templates to disk."""
        if not self._storage_path:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for t in self._templates:
            d = {
                "template_id": t.template_id,
                "server_id": t.server_id,
                "tool_name": t.tool_name,
                "difficulty": t.difficulty,
                "discrimination": t.discrimination,
                "arg_template": t.arg_template,
                "expected_behaviour": t.expected_behaviour,
                "rubric_template": [
                    {"name": r.name, "weight": r.weight, "criteria": r.criteria, "pass_threshold": r.pass_threshold}
                    for r in t.rubric_template
                ],
                "validated": t.validated,
                "created_at": t.created_at.isoformat(),
            }
            data.append(d)
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, template: ProbeTemplate) -> None:
        """Add a template to the library."""
        self._templates.append(template)

    def lookup(
        self,
        tool_name: str,
        target_difficulty: float,
        difficulty_tolerance: float = 0.2,
    ) -> ProbeTemplate | None:
        """Find a matching template by tool name and difficulty.

        Args:
            tool_name: MCP tool name to match.
            target_difficulty: Desired probe difficulty (0-1).
            difficulty_tolerance: Maximum allowed deviation from target.

        Returns:
            Best matching ProbeTemplate, or None if no match found.
            Prefers validated templates when multiple match.
        """
        matches = [
            t for t in self._templates
            if t.tool_name == tool_name
            and abs(t.difficulty - target_difficulty) <= difficulty_tolerance
        ]

        if not matches:
            return None

        # Prefer validated templates, then closest difficulty
        matches.sort(key=lambda t: (-int(t.validated), abs(t.difficulty - target_difficulty)))
        return matches[0]

    def __len__(self) -> int:
        return len(self._templates)
