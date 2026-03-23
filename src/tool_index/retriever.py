"""Online tool retriever — queries the FAISS index at runtime.

Called by Stage 2 to find candidate tools for each subtask.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.tool_index.indexer import ToolRecord

logger = logging.getLogger(__name__)


@dataclass
class ToolCandidate:
    """A tool returned by the retriever with its similarity score."""

    tool_name: str
    server_id: str
    description: str
    similarity_score: float
    capability_tags: list[str]
    parameter_schema: dict[str, Any]
    complexity_estimate: float


class ToolRetriever:
    """Loads a persisted FAISS index and retrieves tools by embedding similarity.

    Args:
        index_dir: Directory containing index.faiss and tools.json.
        embedding_model: Must match the model used at indexing time.
    """

    def __init__(
        self,
        index_dir: str | Path,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "index.faiss"))

        with open(index_dir / "tools.json") as f:
            raw = json.load(f)
        self.tools = [ToolRecord(**r) for r in raw]

        self._encoder = SentenceTransformer(embedding_model)
        logger.info("Loaded tool index with %d tools", len(self.tools))

    def add_tools_at_runtime(self, tools: list[ToolRecord]) -> int:
        """Dynamically add tools to the FAISS index at runtime.

        Encodes tool descriptions, appends embeddings to the existing index,
        and extends the tools list in parallel.

        Args:
            tools: List of ToolRecord objects to add.

        Returns:
            Number of tools successfully added.
        """
        if not tools:
            return 0

        texts = [f"{t.tool_name}: {t.description}" for t in tools]
        embeddings = self._encoder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.tools.extend(tools)

        logger.info("Added %d tools at runtime (total: %d)", len(tools), len(self.tools))
        return len(tools)

    def build_ephemeral_index(
        self, tools: list[ToolRecord]
    ) -> tuple[faiss.Index, list[ToolRecord]] | None:
        """Build a standalone FAISS index for temporary tools.

        Creates a small IndexFlatIP from the given tools without mutating
        the base index or tools list. The caller owns the returned objects
        and they are garbage-collected when no longer referenced.

        Args:
            tools: ToolRecord objects to index ephemerally.

        Returns:
            (index, tools) tuple, or None if *tools* is empty.
        """
        if not tools:
            return None

        texts = [f"{t.tool_name}: {t.description}" for t in tools]
        embeddings = self._encoder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        logger.info("Built ephemeral index with %d tools", len(tools))
        return index, list(tools)

    def retrieve(
        self,
        query: str,
        *,
        candidate_server_ids: set[str] | None = None,
        tag_filter: list[str] | None = None,
        k: int = 20,
        extra_index: tuple[faiss.Index, list[ToolRecord]] | None = None,
    ) -> list[ToolCandidate]:
        """Retrieve the top-k tools matching a query.

        Args:
            query: Natural language subtask description.
            candidate_server_ids: If set, only return tools from these servers.
            tag_filter: If set, pre-filter to tools with at least one matching tag.
            k: Number of results to return.
            extra_index: Optional ephemeral (index, tools) pair to search
                alongside the base index. Results are merged by score.

        Returns:
            List of ToolCandidate sorted by descending similarity.
        """
        # Embed the query
        vec = self._encoder.encode([query], show_progress_bar=False)
        vec = np.array(vec, dtype=np.float32)
        faiss.normalize_L2(vec)

        search_k = min(k * 5, len(self.tools))
        scores, indices = self.index.search(vec, search_k)

        # Collect (score, ToolRecord) pairs from base index
        raw_hits: list[tuple[float, ToolRecord]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            raw_hits.append((float(score), self.tools[idx]))

        # Search the ephemeral index if provided
        if extra_index is not None:
            ep_index, ep_tools = extra_index
            ep_k = min(k * 5, len(ep_tools))
            ep_scores, ep_indices = ep_index.search(vec, ep_k)
            for score, idx in zip(ep_scores[0], ep_indices[0]):
                if idx < 0:
                    continue
                raw_hits.append((float(score), ep_tools[idx]))

        # Sort by descending similarity and apply filters
        raw_hits.sort(key=lambda h: h[0], reverse=True)

        raw_server_ids = {t.server_id for _, t in raw_hits}
        logger.debug(
            "Raw hits: %d (servers: %s), filtering for: %s",
            len(raw_hits), raw_server_ids, candidate_server_ids,
        )

        results: list[ToolCandidate] = []
        for score, tool in raw_hits:
            # Server filter
            if candidate_server_ids and tool.server_id not in candidate_server_ids:
                continue

            # Tag filter
            if tag_filter and not any(t in tool.capability_tags for t in tag_filter):
                continue

            results.append(
                ToolCandidate(
                    tool_name=tool.tool_name,
                    server_id=tool.server_id,
                    description=tool.description,
                    similarity_score=score,
                    capability_tags=tool.capability_tags,
                    parameter_schema=tool.parameter_schema,
                    complexity_estimate=tool.complexity_estimate,
                )
            )

            if len(results) >= k:
                break

        logger.debug("After filter: %d/%d hits survived", len(results), len(raw_hits))
        logger.info(
            "Retrieved %d tools for query: %s",
            len(results),
            query[:60],
        )
        return results
