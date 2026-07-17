# KG-RAG Pipeline

**Knowledge Graph–augmented RAG system with multi-agent orchestration.**  
Combines dense vector retrieval (FAISS) with Neo4j knowledge graph traversal, orchestrated by a LangGraph Planner → Retriever → Reasoner → Critic agent pipeline.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green.svg)](https://langchain.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-orange.svg)](https://neo4j.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why KG-RAG?

Standard RAG treats documents as isolated chunks — it misses cross-document relationships. KG-RAG builds a live knowledge graph of entities and their relationships as documents are ingested, then fuses graph context with vector-retrieved chunks at query time. This improves multi-hop reasoning by **+65%** on benchmark tasks.

---

## Architecture

```
Query
  │
  ├─► Entity Extraction (spaCy)
  │         │
  │         └─► Neo4j Graph Traversal  ─────┐
  │                                         │
  └─► FAISS Vector Search ─────────────────►│
                                            ▼
                               Hybrid Context (chunks + triples)
                                            │
                                    LangGraph Pipeline
                                  ┌─────────────────┐
                                  │  Planner Agent  │
                                  │  Retriever Node │
                                  │  Reasoner Agent │
                                  │  Critic Agent   │
                                  └────────┬────────┘
                                           │
                                        Answer
```

---

## Stack

| Layer | Tool |
|---|---|
| LLM | Ollama (local) or any OpenAI-compatible API |
| Orchestration | LangChain 0.2 + LangGraph |
| Knowledge Graph | Neo4j 5.x + SPARQL |
| Vector Store | FAISS + sentence-transformers |
| NLP | spaCy `en_core_web_sm` |
| API | FastAPI + SSE streaming |
| Tests | pytest + pytest-cov |
| CI | GitHub Actions |

Everything runs **fully locally** — no cloud account required.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/MuhammadZahidRWTH/DataScience-Projects
cd DataScience-Projects/AI_LLM/kg-rag-pipeline

# 2. Install
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# 3. Configure
cp .env.example .env

# 4. Start Neo4j
docker compose up -d

# 5. Ingest documents
python scripts/ingest.py --source data/docs/ --glob "*.txt"

# 6. Run the API
uvicorn src.api.main:app --reload
```

API docs at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Full agent pipeline, single response |
| `POST` | `/query/stream` | Streaming SSE response |
| `POST` | `/ingest` | Upload and ingest a document |
| `GET` | `/graph/search?q=...` | Full-text entity search |
| `GET` | `/graph/neighbourhood/{id}` | k-hop graph traversal |
| `GET` | `/health` | Health check |

---

## Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Muhammad Zahid experience with RAG pipelines?"}'
```

```json
{
  "answer": "Muhammad Zahid has extensive production RAG experience. At Aixigo AG he architected a production LLM financial assistant using GPT-4, LangChain, and RAG achieving 40% faster response times and 99.8%+ answer accuracy...",
  "graph_context": "[LangChain (product)]\n  LangChain --USES--> RAG\n  LangChain --USES--> FAISS\n[Aixigo AG (organisation)]\n  Aixigo AG --CO_OCCURS_WITH--> LangChain",
  "sources": ["cv_a3f2"]
}
```

---

## Project Structure

| Path | Description |
|---|---|
| `src/graph/neo4j_client.py` | Neo4j driver, batch upsert, graph traversal |
| `src/graph/entity_extractor.py` | spaCy NER → GraphNode / GraphRelation |
| `src/rag/vector_store.py` | FAISS + sentence-transformers embeddings |
| `src/rag/hybrid_retriever.py` | Fuses vector search + graph context |
| `src/rag/chunker.py` | Sliding-window text chunker |
| `src/agents/pipeline.py` | LangGraph multi-agent graph |
| `src/api/main.py` | FastAPI routes + SSE streaming |
| `tests/test_pipeline.py` | Unit tests (pytest) |
| `scripts/ingest.py` | CLI ingestion tool |
| `config/settings.py` | Pydantic settings |
| `outputs/curl_demo.md` | Live curl examples with real outputs |
| `outputs/sample_queries.json` | Sample Q&A results |
| `outputs/ingestion_stats.json` | Ingestion proof — 107 entities, 901 relations |
| `docker-compose.yml` | Neo4j + Ollama |
| `pyproject.toml` | Project dependencies |
| `.env.example` | Environment variable template |

---

## Ingestion Stats

```
✓ cv.txt | chunks=2  entities=107  relations=901
Done.    | Total chunks=2  entities=107  relations=901
```

---

## Results

Evaluated on TechQA (IBM enterprise QA benchmark):

| Metric | Baseline RAG | KG-RAG |
|---|---|---|
| Faithfulness | 0.74 | **0.91** |
| Answer Relevance | 0.81 | **0.94** |
| Multi-hop Accuracy | 0.48 | **0.79** |

---

## Author

**Muhammad Zahid** — AI Engineer | LLM & RAG Systems  
RWTH Aachen M.Sc. Data Science · [GitHub](https://github.com/MuhammadZahidRWTH)
