"""
src/agents/pipeline.py
-----------------------
LangGraph multi-agent pipeline:

  Planner → Retriever → Reasoner → Critic → Answer

Each node is a separate agent with a focused role.
The graph runs synchronously (invoke) or as a stream (stream).
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from src.rag.hybrid_retriever import HybridRetriever, RetrievalResult

logger = logging.getLogger(__name__)


# ── Shared state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    plan: str
    retrieval: RetrievalResult | None
    draft_answer: str
    critique: str
    final_answer: str
    iterations: int


# ── Prompts ───────────────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a planning agent. Given a user question, decompose it into
a clear retrieval plan: what facts are needed, what entities are central, what relationships
matter. Be concise (3-5 bullet points). Output only the plan."""

REASONER_PROMPT = """You are an expert reasoner. Use the provided context (document chunks
and knowledge graph triples) to answer the question. Be precise, cite document numbers [1],
[2]... when using chunk content. If graph triples are relevant, weave them in naturally.
Do not hallucinate facts not present in the context."""

CRITIC_PROMPT = """You are a critic agent. Review the draft answer against the retrieved
context. Check for:
- Unsupported claims (not in context)
- Missing key facts from the context
- Logical inconsistencies

If the answer is good, output: APPROVED
If it needs revision, output: REVISE: <short reason>"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def make_planner(llm: BaseChatModel):
    def planner(state: AgentState) -> AgentState:
        logger.debug("[Planner] query=%s", state["query"])
        response = llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=state["query"]),
        ])
        state["plan"] = response.content
        return state
    return planner


def make_retriever(retriever: HybridRetriever):
    def retrieve(state: AgentState) -> AgentState:
        logger.debug("[Retriever] running hybrid retrieval")
        result = retriever.retrieve(state["query"])
        state["retrieval"] = result
        return state
    return retrieve


def make_reasoner(llm: BaseChatModel):
    def reasoner(state: AgentState) -> AgentState:
        logger.debug("[Reasoner] generating answer")
        context = state["retrieval"].combined_context if state["retrieval"] else ""
        messages = [
            SystemMessage(content=REASONER_PROMPT),
            HumanMessage(content=(
                f"Question: {state['query']}\n\n"
                f"Plan:\n{state['plan']}\n\n"
                f"Context:\n{context}"
            )),
        ]
        response = llm.invoke(messages)
        state["draft_answer"] = response.content
        return state
    return reasoner


def make_critic(llm: BaseChatModel):
    def critic(state: AgentState) -> AgentState:
        logger.debug("[Critic] reviewing draft")
        context = state["retrieval"].combined_context if state["retrieval"] else ""
        messages = [
            SystemMessage(content=CRITIC_PROMPT),
            HumanMessage(content=(
                f"Question: {state['query']}\n\n"
                f"Context:\n{context}\n\n"
                f"Draft answer:\n{state['draft_answer']}"
            )),
        ]
        response = llm.invoke(messages)
        state["critique"] = response.content
        state["iterations"] = state.get("iterations", 0) + 1
        return state
    return critic


def route_after_critic(state: AgentState) -> str:
    """Route back to reasoner for revision, or end if approved / max iterations."""
    if state["iterations"] >= 2:
        state["final_answer"] = state["draft_answer"]
        return "end"
    if state["critique"].strip().upper().startswith("APPROVED"):
        state["final_answer"] = state["draft_answer"]
        return "end"
    return "reasoner"   # retry


def finalise(state: AgentState) -> AgentState:
    if not state.get("final_answer"):
        state["final_answer"] = state.get("draft_answer", "No answer generated.")
    return state


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_pipeline(llm: BaseChatModel, retriever: HybridRetriever) -> Any:
    """
    Build and compile the LangGraph pipeline.

    Returns a compiled graph that accepts AgentState and can be invoked with:
        graph.invoke({"query": "...", "retrieval": None, "plan": "",
                      "draft_answer": "", "critique": "", "final_answer": "",
                      "iterations": 0})
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner",   make_planner(llm))
    graph.add_node("retriever", make_retriever(retriever))
    graph.add_node("reasoner",  make_reasoner(llm))
    graph.add_node("critic",    make_critic(llm))
    graph.add_node("finalise",  finalise)

    graph.set_entry_point("planner")
    graph.add_edge("planner",   "retriever")
    graph.add_edge("retriever", "reasoner")
    graph.add_edge("reasoner",  "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"reasoner": "reasoner", "end": "finalise"},
    )
    graph.add_edge("finalise", END)

    return graph.compile()
