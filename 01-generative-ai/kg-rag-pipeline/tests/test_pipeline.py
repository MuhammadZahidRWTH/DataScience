"""tests/test_pipeline.py — Unit tests for core pipeline components"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph.entity_extractor import EntityExtractor, _classify_relation, _uid
from src.rag.chunker import chunk_text
from src.rag.vector_store import Chunk


# ── Chunker ───────────────────────────────────────────────────────────────────

class TestChunker:
    def test_short_text_returns_single_chunk(self):
        text = "Hello world this is a short document."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_long_text_is_split(self):
        text = " ".join(["word"] * 1200)
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) > 1

    def test_overlap_creates_shared_words(self):
        text = " ".join([str(i) for i in range(100)])
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        # Each consecutive pair should share some words
        for a, b in zip(chunks, chunks[1:]):
            words_a = set(a.split())
            words_b = set(b.split())
            assert len(words_a & words_b) > 0

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=512)
        assert chunks == [""]


# ── Entity UID ────────────────────────────────────────────────────────────────

class TestUID:
    def test_same_inputs_same_uid(self):
        assert _uid("OpenAI", "organisation", "doc1") == _uid("OpenAI", "organisation", "doc1")

    def test_case_insensitive(self):
        assert _uid("OpenAI", "organisation", "doc1") == _uid("openai", "organisation", "doc1")

    def test_different_docs_different_uid(self):
        assert _uid("OpenAI", "organisation", "doc1") != _uid("OpenAI", "organisation", "doc2")


# ── Relation classifier ───────────────────────────────────────────────────────

class TestRelationClassifier:
    def test_acquired(self):
        assert _classify_relation("Microsoft acquired Activision.") == "ACQUIRED"

    def test_founded(self):
        assert _classify_relation("She founded the startup in 2018.") == "FOUNDED"

    def test_uses(self):
        assert _classify_relation("The system uses LangChain under the hood.") == "USES"

    def test_default(self):
        assert _classify_relation("Apple and Google both released updates.") == "CO_OCCURS_WITH"


# ── EntityExtractor ───────────────────────────────────────────────────────────

class TestEntityExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        return EntityExtractor()

    def test_extracts_org(self, extractor):
        result = extractor.extract("OpenAI is headquartered in San Francisco.", doc_id="test1")
        names = [n.properties["name"] for n in result.nodes]
        assert any("OpenAI" in n for n in names)

    def test_no_duplicate_nodes(self, extractor):
        text = "Google and Google are both technology companies. Google is big."
        result = extractor.extract(text, doc_id="test2")
        google_nodes = [n for n in result.nodes if "Google" in n.properties["name"]]
        assert len(google_nodes) == 1

    def test_returns_extraction_result(self, extractor):
        from src.graph.entity_extractor import ExtractionResult
        result = extractor.extract("Tesla was founded by Elon Musk.", doc_id="test3")
        assert isinstance(result, ExtractionResult)


# ── VectorStore ───────────────────────────────────────────────────────────────

class TestVectorStore:
    def test_search_returns_results(self, tmp_path):
        from src.rag.vector_store import VectorStore
        vs = VectorStore(model_name="all-MiniLM-L6-v2", index_path=str(tmp_path))
        chunks = [
            Chunk("c1", "The RAG pipeline uses FAISS for vector search.", "doc1", {}),
            Chunk("c2", "Neo4j stores the knowledge graph.", "doc2", {}),
            Chunk("c3", "LangGraph orchestrates multi-agent workflows.", "doc3", {}),
        ]
        vs.add_chunks(chunks)
        results = vs.search("knowledge graph", top_k=2)
        assert len(results) == 2
        texts = [r[0].text for r in results]
        assert any("Neo4j" in t for t in texts)

    def test_empty_search(self, tmp_path):
        from src.rag.vector_store import VectorStore
        vs = VectorStore(model_name="all-MiniLM-L6-v2", index_path=str(tmp_path))
        results = vs.search("anything", top_k=5)
        assert results == []
