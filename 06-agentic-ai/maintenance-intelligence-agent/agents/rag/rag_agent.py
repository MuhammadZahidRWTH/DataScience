# agents/rag/rag_agent.py
# RAG Knowledge Agent — fourth agent in the MaintenanceGPT pipeline
#
# Responsibilities:
#   1. Build FAISS index over machine maintenance manuals
#   2. Answer maintenance questions using hybrid retrieval + Ollama
#   3. Provide procedure recommendations based on diagnosed faults
#   4. Ground LLM answers in actual machine documentation
#
# Design rationale:
#   - RAG over enterprise knowledge bases is core IAIS deployment pattern
#   - Sovereign: manuals stay local, no data sent to cloud
#   - Combines vector retrieval with structured reasoning

import json
from pathlib import Path
from typing import TypedDict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    DATA_MANUALS,
    DATA_PROCESSED,
    FAISS_INDEX_PATH,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
    EMBEDDING_MODEL,
)


# ── Agent State ────────────────────────────────────────────
class RAGState(TypedDict):
    diagnoses: list[dict]
    rag_responses: list[dict]
    vectorstore: object
    summary: str
    errors: list[str]


# ── Build FAISS Index ──────────────────────────────────────
def build_knowledge_index(state: RAGState) -> RAGState:
    """
    Load maintenance manuals and build FAISS vector index.
    Uses sentence-transformers for local embeddings — no API key needed.
    """
    errors = []

    try:
        manual_path = DATA_MANUALS / "machine_manuals.txt"

        loader = TextLoader(str(manual_path), encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n===", "\n\n", "\n", " "],
        )
        chunks = splitter.split_documents(documents)
        print(f"  [RAG] Manual split into {len(chunks)} chunks")

        # local embeddings — sovereign, no cloud
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)

        # persist index for reuse
        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(FAISS_INDEX_PATH))
        print(f"  [RAG] FAISS index built and saved — {len(chunks)} vectors")

        return {**state, "vectorstore": vectorstore, "errors": errors}

    except Exception as e:
        errors.append(f"Index build failed: {e}")
        print(f"  [RAG] ERROR: {e}")
        return {**state, "vectorstore": None, "errors": errors}


# ── Load Diagnoses ─────────────────────────────────────────
def load_diagnoses(state: RAGState) -> RAGState:
    """Load diagnosed faults to generate targeted RAG queries."""
    try:
        with open(DATA_PROCESSED / "diagnoses.json") as f:
            diagnoses = json.load(f)
        print(f"  [RAG] Loaded {len(diagnoses)} diagnoses")
        return {**state, "diagnoses": diagnoses}
    except Exception as e:
        state["errors"].append(f"Diagnoses load failed: {e}")
        return {**state, "diagnoses": []}


# ── RAG Query per Fault ────────────────────────────────────
def query_knowledge_base(state: RAGState) -> RAGState:
    """
    For each diagnosed fault with LLM explanation,
    query the manual for specific maintenance procedure.
    """
    vectorstore = state["vectorstore"]
    diagnoses = state["diagnoses"]

    if vectorstore is None or not diagnoses:
        return {**state, "rag_responses": []}

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    # only query for actual faults with LLM explanations
    fault_diagnoses = [
        d for d in diagnoses
        if d.get("predicted_fault") not in ("none", "unknown")
        and "llm_explanation" in d
    ]

    print(f"  [RAG] Querying knowledge base for {len(fault_diagnoses)} fault diagnoses...")

    rag_responses = []

    for diag in fault_diagnoses:
        machine = diag["machine_id"]
        fault = diag["predicted_fault"]
        sensor = diag["sensor"]

        # build targeted query
        query = f"{machine} {fault} {sensor} maintenance procedure action required"

        # retrieve relevant manual chunks
        try:
            docs = vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""You are a maintenance engineer at a German manufacturing plant.
Use the machine manual excerpts below to answer the maintenance question.
Only use information from the manual. Be specific and concise.

MANUAL EXCERPTS:
{context}

QUESTION: 
Machine {machine} has a diagnosed fault: {fault} on {sensor} sensor.
What is the exact maintenance procedure and immediate action required?

Answer in 3 bullet points maximum."""

            response = llm.invoke([HumanMessage(content=prompt)])

            rag_responses.append({
                "machine_id": machine,
                "fault": fault,
                "sensor": sensor,
                "manual_procedure": response.content.strip(),
                "sources": [doc.page_content[:150] for doc in docs[:2]],
            })

        except Exception as e:
            rag_responses.append({
                "machine_id": machine,
                "fault": fault,
                "sensor": sensor,
                "manual_procedure": f"Procedure lookup failed: {e}",
                "sources": [],
            })

    print(f"  [RAG] Generated {len(rag_responses)} procedure recommendations")
    return {**state, "rag_responses": rag_responses}


# ── Save RAG Responses ─────────────────────────────────────
def save_rag_responses(state: RAGState) -> RAGState:
    """Save RAG responses for Report Agent."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(DATA_PROCESSED / "rag_responses.json", "w") as f:
        json.dump(state["rag_responses"], f, indent=2)

    print(f"  [RAG] Saved {len(state['rag_responses'])} responses to data/processed/")
    return state


# ── Summary ────────────────────────────────────────────────
def generate_summary(state: RAGState) -> RAGState:
    n = len(state["rag_responses"])
    summary = f"RAG complete. {n} maintenance procedures retrieved from machine manuals."
    print(f"  [RAG] {summary}")
    return {**state, "summary": summary}


# ── Run standalone ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== MaintenanceGPT — RAG Knowledge Agent ===\n")

    state: RAGState = {
        "diagnoses": [],
        "rag_responses": [],
        "vectorstore": None,
        "summary": "",
        "errors": [],
    }

    state = build_knowledge_index(state)
    state = load_diagnoses(state)
    state = query_knowledge_base(state)
    state = save_rag_responses(state)
    state = generate_summary(state)

    print("\n=== RAG Agent Complete ===")
    print(f"Summary: {state['summary']}")

    if state["rag_responses"]:
        print("\nSample procedure:")
        r = state["rag_responses"][0]
        print(f"  Machine: {r['machine_id']} | Fault: {r['fault']}")
        print(f"  Procedure:\n{r['manual_procedure']}")
