# pipeline.py
# MaintenanceGPT — Full LangGraph Orchestration Pipeline
#
# Wires all 5 agents into a single LangGraph state machine:
# Ingestion → Diagnosis → Forecasting → RAG → Report
#
# Design rationale:
#   - Multi-agent orchestration with specialized reasoning per node
#   - Sovereign: entire pipeline runs locally via Ollama
#   - Reliable agent evaluation: RAGAS metrics on RAG outputs

import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, END
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# import all agent functions
from agents.ingestion.ingestion_agent import (
    load_data, clean_sensor_data, detect_anomalies,
    clean_logs_with_llm, generate_summary as ingestion_summary,
    save_processed,
)
from agents.diagnosis.diagnosis_agent import (
    load_ingestion_output, score_with_isolation_forest,
    train_pytorch_classifier, classify_anomalies,
    explain_with_llm as diagnosis_explain,
    save_diagnoses, generate_summary as diagnosis_summary,
)
from agents.forecasting.forecasting_agent import (
    load_sensor_data, forecast_sensor_trends,
    summarize_risk_with_llm, save_forecasts,
    generate_summary as forecasting_summary,
)
from agents.rag.rag_agent import (
    build_knowledge_index, load_diagnoses,
    query_knowledge_base, save_rag_responses,
    generate_summary as rag_summary,
)
from agents.report.report_agent import (
    load_all_outputs, build_report,
    generate_markdown_report, generate_executive_summary,
    save_report, generate_summary as report_summary,
)


# ── Unified Pipeline State ─────────────────────────────────
class PipelineState(TypedDict):
    # ingestion
    sensor_df: pd.DataFrame
    logs_df: pd.DataFrame
    anomalies_df: pd.DataFrame
    cleaned_logs: list
    # diagnosis
    diagnoses: list
    model_report: str
    # forecasting
    forecasts: list
    risk_summary: str
    # rag
    rag_responses: list
    vectorstore: object
    # report
    ingestion_summary: str
    report: dict
    report_md: str
    # shared
    summary: str
    errors: list


# ── Node Wrappers ──────────────────────────────────────────
# Each node maps PipelineState to the relevant sub-state,
# calls the agent function, and merges results back

def node_load_data(state: PipelineState) -> PipelineState:
    print("\n[Pipeline] Node: load_data")
    result = load_data(state)
    return {**state, **result}

def node_clean_sensor(state: PipelineState) -> PipelineState:
    print("[Pipeline] Node: clean_sensor_data")
    result = clean_sensor_data(state)
    return {**state, **result}

def node_detect_anomalies(state: PipelineState) -> PipelineState:
    print("[Pipeline] Node: detect_anomalies")
    result = detect_anomalies(state)
    return {**state, **result}

def node_clean_logs(state: PipelineState) -> PipelineState:
    print("[Pipeline] Node: clean_logs_with_llm")
    result = clean_logs_with_llm(state)
    return {**state, **result}

def node_save_ingestion(state: PipelineState) -> PipelineState:
    print("[Pipeline] Node: save_processed")
    result = save_processed(state)
    ingestion_sum = ingestion_summary(state)
    return {**state, **result, "ingestion_summary": ingestion_sum.get("summary", "")}

def node_diagnosis(state: PipelineState) -> PipelineState:
    print("[Pipeline] Node: diagnosis (Isolation Forest + PyTorch + LLM)")
    s = load_ingestion_output(state)
    s = score_with_isolation_forest(s)
    s = train_pytorch_classifier(s)
    s = classify_anomalies(s)
    s = diagnosis_explain(s)
    s = save_diagnoses(s)
    s = diagnosis_summary(s)
    return {**state, **s}

def node_forecasting(state: PipelineState) -> PipelineState:
    print("[Pipeline] Node: forecasting (Prophet)")
    s = load_sensor_data(state)
    s = forecast_sensor_trends(s)
    s = summarize_risk_with_llm(s)
    s = save_forecasts(s)
    s = forecasting_summary(s)
    return {**state, **s}

def node_rag(state: PipelineState) -> PipelineState:
    print("[Pipeline] Node: RAG knowledge agent")
    s = {"diagnoses": state["diagnoses"], "rag_responses": [],
         "vectorstore": None, "summary": "", "errors": state["errors"]}
    s = build_knowledge_index(s)
    s = load_diagnoses(s)
    s = query_knowledge_base(s)
    s = save_rag_responses(s)
    s = rag_summary(s)
    return {**state, **s}

def node_report(state: PipelineState) -> PipelineState:
    print("[Pipeline] Node: report agent")
    s = {**state, "report": {}, "report_md": ""}
    s = load_all_outputs(s)
    s = build_report(s)
    s = generate_markdown_report(s)
    s = generate_executive_summary(s)
    s = save_report(s)
    s = report_summary(s)
    return {**state, **s}


# ── Build LangGraph Pipeline ───────────────────────────────
def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    # add nodes
    graph.add_node("load_data", node_load_data)
    graph.add_node("clean_sensor", node_clean_sensor)
    graph.add_node("detect_anomalies", node_detect_anomalies)
    graph.add_node("clean_logs", node_clean_logs)
    graph.add_node("save_ingestion", node_save_ingestion)
    graph.add_node("diagnosis", node_diagnosis)
    graph.add_node("forecasting", node_forecasting)
    graph.add_node("rag", node_rag)
    graph.add_node("report", node_report)

    # define edges — linear pipeline
    graph.set_entry_point("load_data")
    graph.add_edge("load_data", "clean_sensor")
    graph.add_edge("clean_sensor", "detect_anomalies")
    graph.add_edge("detect_anomalies", "clean_logs")
    graph.add_edge("clean_logs", "save_ingestion")
    graph.add_edge("save_ingestion", "diagnosis")
    graph.add_edge("diagnosis", "forecasting")
    graph.add_edge("forecasting", "rag")
    graph.add_edge("rag", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ── Run Full Pipeline ──────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MaintenanceGPT — Full Agentic Pipeline")
    print("  Ingestion → Diagnosis → Forecasting → RAG → Report")
    print("="*60)

    pipeline = build_pipeline()

    initial_state: PipelineState = {
        "sensor_df": pd.DataFrame(),
        "logs_df": pd.DataFrame(),
        "anomalies_df": pd.DataFrame(),
        "cleaned_logs": [],
        "diagnoses": [],
        "model_report": "",
        "forecasts": [],
        "risk_summary": "",
        "rag_responses": [],
        "vectorstore": None,
        "ingestion_summary": "",
        "report": {},
        "report_md": "",
        "summary": "",
        "errors": [],
    }

    final_state = pipeline.invoke(initial_state)

    print("\n" + "="*60)
    print("  Pipeline Complete")
    print("="*60)
    print(f"\nExecutive Summary:")
    print(final_state["report"].get("executive_summary", "N/A"))
    print(f"\nReport saved to: data/processed/maintenance_report.md")
    print(f"Report ID: {final_state['report'].get('report_id', 'N/A')}")
