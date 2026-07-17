"""src/rag/chunker.py — Sliding-window text chunker with sentence boundary awareness"""

from __future__ import annotations

import re


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split text into overlapping chunks by approximate token count (word-level).

    Args:
        text: Raw document text.
        chunk_size: Target chunk size in words.
        overlap: Word overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks
