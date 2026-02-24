"""Offline tool indexer — builds a FAISS index over individual MCP tools.

Runs once when new MCP servers are added. For each tool, embeds
"{tool.name}: {tool.description}" using sentence-transformers and stores
metadata alongside the index.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class ToolRecord:
    """Metadata for a single indexed MCP tool."""

    tool_name: str
    server_id: str
    description: str
    capability_tags: list[str] = field(default_factory=list)
    parameter_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    complexity_estimate: float = 0.5


class ToolIndexer:
    """Builds and persists a FAISS index for MCP tools.

    Args:
        embedding_model: Name of the sentence-transformers model.
        nlist: Number of IVF clusters for the FAISS index.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        nlist: int = 100,
    ) -> None:
        self.embedding_model_name = embedding_model
        self.nlist = nlist
        self._encoder = SentenceTransformer(embedding_model)
        self._dimension = self._encoder.get_sentence_embedding_dimension()
        self.tools: list[ToolRecord] = []
        self.index: faiss.Index | None = None

    def add_tools(self, tools: list[ToolRecord]) -> None:
        """Add tool records to the indexer (call build_index after)."""
        self.tools.extend(tools)

    def build_index(self) -> None:
        """Build the FAISS index from all added tools.

        Uses a flat index for small collections (<1000 tools) and
        IVF for larger ones.
        """
        if not self.tools:
            raise ValueError("No tools to index")

        texts = [f"{t.tool_name}: {t.description}" for t in self.tools]
        embeddings = self._encoder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        n = len(self.tools)
        if n < self.nlist * 10:
            # Too few vectors for IVF — use flat index
            self.index = faiss.IndexFlatIP(self._dimension)
        else:
            quantiser = faiss.IndexFlatIP(self._dimension)
            self.index = faiss.IndexIVFFlat(
                quantiser, self._dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)

        self.index.add(embeddings)
        logger.info("Built FAISS index with %d tools (dim=%d)", n, self._dimension)

    def save(self, directory: str | Path) -> None:
        """Save the FAISS index and tool metadata to disk.

        Args:
            directory: Directory to write index.faiss and tools.json.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise ValueError("Index not built yet — call build_index first")

        faiss.write_index(self.index, str(directory / "index.faiss"))

        metadata = [asdict(t) for t in self.tools]
        with open(directory / "tools.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved index to %s", directory)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string for retrieval.

        Args:
            text: The query text to embed.

        Returns:
            Normalised embedding vector of shape (1, dim).
        """
        vec = self._encoder.encode([text], show_progress_bar=False)
        vec = np.array(vec, dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec
