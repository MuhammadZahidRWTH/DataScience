"""src/rag/hybrid_retriever.py — Fuses FAISS vector search with Neo4j graph context"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.graph.entity_extractor import EntityExtractor
from src.graph.neo4j_client import Neo4jClient
from src.rag.vector_store import Chunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[tuple[Chunk, float]]
    graph_context: str
    combined_context: str


class HybridRetriever:
    """
    Combines dense vector retrieval (FAISS) with Neo4j knowledge graph traversal.

    Pipeline:
      1. Vector search → top-k semantically similar chunks.
      2. Entity extraction from query → graph neighbourhood lookup.
      3. Format combined context string for the LLM prompt.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        neo4j: Neo4jClient,
        extractor: EntityExtractor,
        top_k: int = 6,
        graph_hops: int = 2,
    ) -> None:
        self.vs = vector_store
        self.neo4j = neo4j
        self.extractor = extractor
        self.top_k = top_k
        self.graph_hops = graph_hops

    def retrieve(self, query: str) -> RetrievalResult:
        # 1. Vector search
        chunk_results = self.vs.search(query, top_k=self.top_k)

        # 2. Graph context
        graph_context = self._get_graph_context(query)

        # 3. Format
        combined = self._format(chunk_results, graph_context)

        return RetrievalResult(
            query=query,
            chunks=chunk_results,
            graph_context=graph_context,
            combined_context=combined,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_graph_context(self, query: str) -> str:
        extraction = self.extractor.extract(query, doc_id="__query__")
        if not extraction.raw_entities:
            return ""

        lines: list[str] = []
        for ent in extraction.raw_entities[:4]:  # cap to 4 entities
            matches = self.neo4j.fulltext_search(ent["text"], limit=1)
            if not matches:
                continue
            top = matches[0]
            triples = self.neo4j.neighbourhood(top["id"], hops=self.graph_hops, limit=20)
            if triples:
                lines.append(f"[{ent['text']} ({ent['label']})]")
                for t in triples:
                    lines.append(f"  {t['src']} --{t['rtype']}--> {t['tgt']}")

        return "\n".join(lines)

    @staticmethod
    def _format(chunks: list[tuple[Chunk, float]], graph_context: str) -> str:
        parts: list[str] = []

        if graph_context:
            parts.append("=== Knowledge Graph ===")
            parts.append(graph_context)

        if chunks:
            parts.append("=== Retrieved Documents ===")
            for i, (chunk, score) in enumerate(chunks, 1):
                parts.append(f"[{i}] (score={score:.3f}, doc={chunk.doc_id})\n{chunk.text}")

        return "\n\n".join(parts)
