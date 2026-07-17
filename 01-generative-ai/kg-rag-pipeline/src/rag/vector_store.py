"""src/rag/vector_store.py — FAISS vector store with sentence-transformers embeddings"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    metadata: dict[str, Any]


class VectorStore:
    """
    FAISS-based vector store with sentence-transformers embeddings.
    Persists index + metadata to disk so it survives restarts.

    Args:
        model_name: SentenceTransformer model (e.g. 'all-MiniLM-L6-v2').
        index_path: Directory to save/load the FAISS index and metadata.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "data/faiss_index") -> None:
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dim)  # inner-product = cosine on normalised vecs
        self._chunks: list[Chunk] = []

        self._load_if_exists()
        logger.info("VectorStore ready (model=%s, dim=%d, chunks=%d)", model_name, self.dim, len(self._chunks))

    # ── Ingest ────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vecs = self._embed(texts)
        self._index.add(vecs)
        self._chunks.extend(chunks)
        self._save()
        logger.info("Added %d chunks (total=%d)", len(chunks), len(self._chunks))

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        if self._index.ntotal == 0:
            return []
        q_vec = self._embed([query])
        scores, indices = self._index.search(q_vec, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._chunks[idx], float(score)))
        return results

    def embed_text(self, text: str) -> list[float]:
        return self._embed([text])[0].tolist()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        faiss.write_index(self._index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def _load_if_exists(self) -> None:
        idx_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "chunks.pkl"
        if idx_file.exists() and meta_file.exists():
            self._index = faiss.read_index(str(idx_file))
            with open(meta_file, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info("Loaded existing FAISS index (%d vectors)", self._index.ntotal)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype="float32")
