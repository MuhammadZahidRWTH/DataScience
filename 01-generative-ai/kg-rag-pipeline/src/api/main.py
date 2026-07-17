"""src/api/main.py — FastAPI gateway with /query (streaming SSE) and /ingest endpoints"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config.settings import settings
from src.graph.entity_extractor import EntityExtractor
from src.graph.neo4j_client import Neo4jClient
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.vector_store import Chunk, VectorStore


# ── Logging ───────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="KG-RAG Pipeline API",
    description="Knowledge Graph–augmented RAG with multi-agent orchestration",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state (initialised at startup) ────────────────────────────────────

_neo4j: Neo4jClient | None = None
_vector_store: VectorStore | None = None
_retriever: HybridRetriever | None = None
_pipeline = None


@app.on_event("startup")
async def startup() -> None:
    global _neo4j, _vector_store, _retriever, _pipeline

    try:
        _neo4j = Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        _neo4j.setup()
    except Exception as e:
        logger.warning("Neo4j not available: %s", e)
        _neo4j = None

    _vector_store = VectorStore(
        model_name=settings.embedding_model,
        index_path=settings.faiss_index_path,
    )

    extractor = EntityExtractor()

    _retriever = HybridRetriever(
        vector_store=_vector_store,
        neo4j=_neo4j,
        extractor=extractor,
        top_k=settings.final_k,
        graph_hops=settings.top_k_graph_hops,
    )

    # Build LangGraph pipeline
    llm = _build_llm()
    from src.agents.pipeline import build_pipeline
    _pipeline = build_pipeline(llm, _retriever)

    logger.info("startup_complete", backend=settings.llm_backend)


@app.on_event("shutdown")
async def shutdown() -> None:
    if _neo4j:
        _neo4j.close()


def _build_llm():
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=key)
# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    graph_context: str
    sources: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Single-shot Q&A through the full agent pipeline."""
    if not _pipeline:
        raise HTTPException(503, "Pipeline not initialised")

    logger.info("query_received", query=req.query)

    result = await asyncio.to_thread(
        _pipeline.invoke,
        {
            "query": req.query,
            "plan": "",
            "retrieval": None,
            "draft_answer": "",
            "critique": "",
            "final_answer": "",
            "iterations": 0,
        },
    )

    retrieval = result.get("retrieval")
    sources = []
    graph_ctx = ""
    if retrieval:
        sources = list({c.doc_id for c, _ in retrieval.chunks})
        graph_ctx = retrieval.graph_context

    return QueryResponse(
        answer=result["final_answer"],
        graph_context=graph_ctx,
        sources=sources,
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming SSE endpoint — yields tokens as they arrive (Ollama / OpenAI streaming)."""

    async def event_generator():
        retrieval = await asyncio.to_thread(_retriever.retrieve, req.query)
        context = retrieval.combined_context

        # Build prompt
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.agents.pipeline import REASONER_PROMPT

        llm = _build_llm()
        messages = [
            SystemMessage(content=REASONER_PROMPT),
            HumanMessage(content=f"Question: {req.query}\n\nContext:\n{context}"),
        ]

        async for chunk in llm.astream(messages):
            yield {"data": chunk.content}

        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Ingest a text or PDF file into the vector store and knowledge graph."""
    if not _vector_store or not _neo4j:
        raise HTTPException(503, "Services not initialised")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    doc_id = str(uuid.uuid4())[:8]

    # Chunk text (simple sentence-window chunking)
    from src.rag.chunker import chunk_text
    chunks_text = chunk_text(text, chunk_size=512, overlap=64)

    chunks = [
        Chunk(
            chunk_id=f"{doc_id}_{i}",
            text=c,
            doc_id=doc_id,
            metadata={"filename": file.filename, "chunk_index": i},
        )
        for i, c in enumerate(chunks_text)
    ]
    _vector_store.add_chunks(chunks)

    # Extract entities and write to graph
    from src.graph.entity_extractor import EntityExtractor
    extractor = EntityExtractor()
    extraction = extractor.extract(text, doc_id=doc_id)
    _neo4j.upsert_nodes(extraction.nodes)
    _neo4j.upsert_relations(extraction.relations)

    logger.info(
        "ingested",
        doc_id=doc_id,
        filename=file.filename,
        chunks=len(chunks),
        entities=len(extraction.nodes),
        relations=len(extraction.relations),
    )

    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "entities": len(extraction.nodes),
        "relations": len(extraction.relations),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "backend": settings.llm_backend}


@app.get("/graph/search")
async def graph_search(q: str, limit: int = 10):
    """Full-text entity search in the knowledge graph."""
    if not _neo4j:
        raise HTTPException(503, "Neo4j not initialised")
    results = _neo4j.fulltext_search(q, limit=limit)
    return {"entities": results}


@app.get("/graph/neighbourhood/{entity_id}")
async def graph_neighbourhood(entity_id: str, hops: int = 2):
    """Return k-hop neighbourhood of a graph entity."""
    if not _neo4j:
        raise HTTPException(503, "Neo4j not initialised")
    triples = _neo4j.neighbourhood(entity_id, hops=hops)
    return {"entity_id": entity_id, "triples": triples}

