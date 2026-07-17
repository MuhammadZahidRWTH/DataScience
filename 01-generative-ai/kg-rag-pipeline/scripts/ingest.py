#!/usr/bin/env python
"""
scripts/ingest.py
-----------------
CLI tool to ingest documents from a directory into the vector store and knowledge graph.

Usage:
    python scripts/ingest.py --source data/docs/
    python scripts/ingest.py --source data/docs/ --glob "*.txt"
"""

import argparse
import logging
import uuid
from pathlib import Path

from config.settings import settings
from src.graph.entity_extractor import EntityExtractor
from src.graph.neo4j_client import Neo4jClient
from src.rag.chunker import chunk_text
from src.rag.vector_store import Chunk, VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_file(path: Path, vector_store: VectorStore, neo4j: Neo4jClient, extractor: EntityExtractor) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    doc_id = path.stem[:8] + "_" + str(uuid.uuid4())[:4]

    chunks_text = chunk_text(text, chunk_size=512, overlap=64)
    chunks = [
        Chunk(
            chunk_id=f"{doc_id}_{i}",
            text=c,
            doc_id=doc_id,
            metadata={"filename": path.name, "chunk_index": i},
        )
        for i, c in enumerate(chunks_text)
    ]
    vector_store.add_chunks(chunks)

    extraction = extractor.extract(text, doc_id=doc_id)
    neo4j.upsert_nodes(extraction.nodes)
    neo4j.upsert_relations(extraction.relations)

    return {
        "file": path.name,
        "doc_id": doc_id,
        "chunks": len(chunks),
        "entities": len(extraction.nodes),
        "relations": len(extraction.relations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into KG-RAG pipeline")
    parser.add_argument("--source", required=True, help="Directory containing documents")
    parser.add_argument("--glob", default="*.txt", help="File glob pattern (default: *.txt)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    files = list(source.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matching '{args.glob}' in {source}")

    logger.info("Initialising services...")
    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )
    neo4j.setup()

    vector_store = VectorStore(
        model_name=settings.embedding_model,
        index_path=settings.faiss_index_path,
    )
    extractor = EntityExtractor()

    logger.info("Ingesting %d files from %s", len(files), source)

    total_chunks = total_entities = total_relations = 0
    for f in files:
        try:
            result = ingest_file(f, vector_store, neo4j, extractor)
            logger.info(
                "✓ %s | chunks=%d entities=%d relations=%d",
                result["file"], result["chunks"], result["entities"], result["relations"],
            )
            total_chunks += result["chunks"]
            total_entities += result["entities"]
            total_relations += result["relations"]
        except Exception as e:
            logger.error("✗ %s — %s", f.name, e)

    neo4j.close()
    logger.info(
        "Done. Total: chunks=%d entities=%d relations=%d",
        total_chunks, total_entities, total_relations,
    )


if __name__ == "__main__":
    main()
