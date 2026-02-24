#!/usr/bin/env python3
"""Generate initial probe template library from indexed MCP servers.

Uses the back-instruct method: reads tool schemas, generates 2-3 probe templates
per tool at varying difficulty levels, validates against schema, and stores
in the template library.

Usage:
    python scripts/bootstrap_templates.py \
        --servers tests/fixtures/sample_mcp_servers.json \
        --output data/templates.json \
        --api-key $CEREBRAS_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.client import LLMClient
from src.templates.generator import generate_templates_for_tool
from src.templates.library import TemplateLibrary

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap probe template library")
    parser.add_argument(
        "--servers", required=True,
        help="Path to JSON file with MCP server definitions",
    )
    parser.add_argument(
        "--output", default="data/templates.json",
        help="Output path for template library JSON",
    )
    parser.add_argument("--api-key", required=True, help="LLM API key")
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML (defaults to config/default.yaml)",
    )
    parser.add_argument(
        "--difficulty-levels", nargs="+", type=float, default=[0.2, 0.5, 0.8],
        help="Difficulty levels to generate templates at",
    )
    args = parser.parse_args()

    llm = LLMClient.from_config(api_key=args.api_key, config_path=args.config)
    library = TemplateLibrary(storage_path=args.output)

    with open(args.servers) as f:
        servers = json.load(f)

    total_tools = sum(len(s["tools"]) for s in servers)
    logger.info("Generating templates for %d tools across %d servers", total_tools, len(servers))

    generated = 0
    failed = 0

    for server in servers:
        server_id = server["server_id"]
        for tool in server["tools"]:
            logger.info("Generating templates for %s/%s", server_id, tool["tool_name"])
            try:
                templates = generate_templates_for_tool(
                    tool_name=tool["tool_name"],
                    server_id=server_id,
                    tool_description=tool["description"],
                    parameter_schema=tool.get("parameter_schema", {}),
                    llm=llm,
                    difficulty_levels=args.difficulty_levels,
                )
                for t in templates:
                    library.add(t)
                generated += len(templates)
            except Exception as e:
                logger.error("Failed for %s/%s: %s", server_id, tool["tool_name"], e)
                failed += 1

    library.save()

    tokens = llm.total_tokens()
    logger.info(
        "Done: %d templates generated, %d tools failed. Tokens: %s",
        generated, failed, tokens,
    )


if __name__ == "__main__":
    main()
