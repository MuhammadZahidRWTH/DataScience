# agents/report/report_agent.py
# Report Agent — fifth and final agent in the MaintenanceGPT pipeline
#
# Responsibilities:
#   1. Load outputs from all previous agents
#   2. Compile structured maintenance report in JSON + Markdown
#   3. Prioritise actions by severity
#   4. Generate executive summary via Ollama
#
# Design rationale:
#   - Automated knowledge work report generation
#   Automates knowledge work report generation.
#   - Sovereign: entire report generated locally, no cloud

import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    DATA_PROCESSED,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)


# ── Agent State ────────────────────────────────────────────
class ReportState(TypedDict):
    ingestion_summary: str
    diagnoses: list[dict]
    forecasts: list[dict]
    risk_summary: str
    rag_responses: list[dict]
    report: dict
    report_md: str
    summary: str
    errors: list[str]


# ── Load All Agent Outputs ─────────────────────────────────
def load_all_outputs(state: ReportState) -> ReportState:
    """Load outputs from all previous agents."""
    errors = []

    try:
        with open(DATA_PROCESSED / "diagnoses.json") as f:
            diagnoses = json.load(f)
        print(f"  [Report] Loaded {len(diagnoses)} diagnoses")
    except Exception as e:
        diagnoses = []
        errors.append(f"Diagnoses load failed: {e}")

    try:
        with open(DATA_PROCESSED / "forecasts.json") as f:
            forecast_data = json.load(f)
        forecasts = forecast_data.get("forecasts", [])
        risk_summary = forecast_data.get("risk_summary", "")
        print(f"  [Report] Loaded {len(forecasts)} forecasts")
    except Exception as e:
        forecasts = []
        risk_summary = ""
        errors.append(f"Forecasts load failed: {e}")

    try:
        with open(DATA_PROCESSED / "rag_responses.json") as f:
            rag_responses = json.load(f)
        print(f"  [Report] Loaded {len(rag_responses)} RAG responses")
    except Exception as e:
        rag_responses = []
        errors.append(f"RAG responses load failed: {e}")

    try:
        with open(DATA_PROCESSED / "cleaned_logs.json") as f:
            cleaned_logs = json.load(f)
        ingestion_summary = (
            f"Processed {len(cleaned_logs)} maintenance log entries. "
            f"Machines monitored: CNC_Mill_01, Hydraulic_Press_02, Conveyor_Motor_03."
        )
    except Exception as e:
        ingestion_summary = "Ingestion data unavailable."
        errors.append(f"Logs load failed: {e}")

    return {
        **state,
        "diagnoses": diagnoses,
        "forecasts": forecasts,
        "risk_summary": risk_summary,
        "rag_responses": rag_responses,
        "ingestion_summary": ingestion_summary,
        "errors": errors,
    }


# ── Build Structured Report ────────────────────────────────
def build_report(state: ReportState) -> ReportState:
    """
    Compile structured JSON report from all agent outputs.
    Prioritises actions by severity.
    """
    diagnoses = state["diagnoses"]
    forecasts = state["forecasts"]
    rag_responses = state["rag_responses"]

    # fault summary
    fault_counts = {}
    for d in diagnoses:
        ft = d.get("predicted_fault", "unknown")
        fault_counts[ft] = fault_counts.get(ft, 0) + 1

    # high priority: faults with LLM explanation
    high_priority = [
        d for d in diagnoses
        if d.get("predicted_fault") not in ("none", "unknown")
        and "llm_explanation" in d
    ]

    # medium priority: faults without explanation
    medium_priority = [
        d for d in diagnoses
        if d.get("predicted_fault") not in ("none", "unknown")
        and "llm_explanation" not in d
    ]

    # at-risk forecasts
    at_risk_forecasts = [
        f for f in forecasts
        if f.get("at_risk")
    ]

    # build action items
    action_items = []

    for d in high_priority:
        action_items.append({
            "priority": "HIGH",
            "machine": d["machine_id"],
            "fault": d["predicted_fault"],
            "sensor": d["sensor"],
            "confidence": d.get("confidence"),
            "diagnosis": d.get("llm_explanation", ""),
            "procedure": next(
                (r["manual_procedure"] for r in rag_responses
                 if r["machine_id"] == d["machine_id"]
                 and r["fault"] == d["predicted_fault"]),
                "See machine manual."
            ),
        })

    for d in medium_priority[:5]:  # cap to top 5
        action_items.append({
            "priority": "MEDIUM",
            "machine": d["machine_id"],
            "fault": d["predicted_fault"],
            "sensor": d["sensor"],
            "confidence": d.get("confidence"),
            "diagnosis": "Fault detected — manual review recommended.",
            "procedure": "Refer to machine maintenance manual.",
        })

    report = {
        "report_id": f"MG-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "generated_at": datetime.now().isoformat(),
        "generated_by": "MaintenanceGPT — Sovereign Agentic AI System",
        "pipeline": "Ingestion → Diagnosis → Forecasting → RAG → Report",
        "ingestion_summary": state["ingestion_summary"],
        "fault_summary": fault_counts,
        "total_anomalies": len(diagnoses),
        "high_priority_count": len(high_priority),
        "medium_priority_count": len(medium_priority),
        "forecast_risk_summary": state["risk_summary"],
        "at_risk_machines": len(at_risk_forecasts),
        "action_items": action_items,
    }

    print(f"  [Report] Built report — {len(action_items)} action items "
          f"({len(high_priority)} high, {len(medium_priority[:5])} medium priority)")

    return {**state, "report": report}


# ── Generate Markdown Report ───────────────────────────────
def generate_markdown_report(state: ReportState) -> ReportState:
    """Convert structured report to human-readable Markdown."""
    report = state["report"]
    lines = []

    lines.append(f"# MaintenanceGPT — Maintenance Intelligence Report")
    lines.append(f"**Report ID:** {report['report_id']}")
    lines.append(f"**Generated:** {report['generated_at']}")
    lines.append(f"**System:** {report['generated_by']}")
    lines.append(f"**Pipeline:** {report['pipeline']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- **Total anomalies detected:** {report['total_anomalies']}")
    lines.append(f"- **High priority actions:** {report['high_priority_count']}")
    lines.append(f"- **Medium priority actions:** {report['medium_priority_count']}")
    lines.append(f"- **At-risk machines (48h forecast):** {report['at_risk_machines']}")
    lines.append("")
    lines.append("**Fault breakdown:**")
    for fault, count in report["fault_summary"].items():
        lines.append(f"- {fault}: {count}")
    lines.append("")
    lines.append("**Forecast risk assessment:**")
    lines.append(report["forecast_risk_summary"])
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Action Items")
    lines.append("")

    for i, item in enumerate(report["action_items"], 1):
        priority_emoji = "🔴" if item["priority"] == "HIGH" else "🟡"
        lines.append(f"### {i}. {priority_emoji} [{item['priority']}] {item['machine']} — {item['fault']}")
        lines.append(f"**Sensor:** {item['sensor']} | **Confidence:** {item.get('confidence', 'N/A')}")
        lines.append("")
        lines.append(f"**Diagnosis:** {item['diagnosis']}")
        lines.append("")
        lines.append(f"**Recommended Procedure:**")
        lines.append(item["procedure"])
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Ingestion Summary")
    lines.append(report["ingestion_summary"])
    lines.append("")
    lines.append("---")
    lines.append("*Generated by MaintenanceGPT — Sovereign Agentic AI for Industrial Predictive Maintenance*")
    lines.append("*Runs fully locally via Ollama — GDPR compliant, no cloud dependency*")

    report_md = "\n".join(lines)
    return {**state, "report_md": report_md}


# ── Generate Executive Summary via LLM ────────────────────
def generate_executive_summary(state: ReportState) -> ReportState:
    """Use Ollama to generate a short executive summary."""
    report = state["report"]

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    prompt = f"""You are a maintenance operations manager at a German manufacturing plant.

Summarize this maintenance intelligence report in 3 sentences for plant management.
Focus on: what was found, what needs immediate action, and expected impact if addressed.

Key findings:
- Total anomalies: {report['total_anomalies']}
- High priority faults: {report['high_priority_count']}
- Fault types: {report['fault_summary']}
- Forecast: {report['forecast_risk_summary']}

Be direct and professional. Write for a non-technical plant manager."""

    print(f"  [Report] Generating executive summary via LLM...")

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        executive_summary = response.content.strip()
        state["report"]["executive_summary"] = executive_summary
        print(f"  [Report] Executive summary generated")
    except Exception as e:
        state["report"]["executive_summary"] = f"Summary unavailable: {e}"

    return state


# ── Save Report ────────────────────────────────────────────
def save_report(state: ReportState) -> ReportState:
    """Save report as JSON and Markdown."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(DATA_PROCESSED / "maintenance_report.json", "w") as f:
        json.dump(state["report"], f, indent=2, default=str)

    with open(DATA_PROCESSED / "maintenance_report.md", "w", encoding="utf-8") as f:
        f.write(state["report_md"])

    print(f"  [Report] Saved maintenance_report.json and maintenance_report.md")
    return state


# ── Summary ────────────────────────────────────────────────
def generate_summary(state: ReportState) -> ReportState:
    report = state["report"]
    summary = (
        f"Report complete. "
        f"ID: {report['report_id']}. "
        f"{len(report['action_items'])} action items generated."
    )
    print(f"  [Report] {summary}")
    return {**state, "summary": summary}


# ── Run standalone ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== MaintenanceGPT — Report Agent ===\n")

    state: ReportState = {
        "ingestion_summary": "",
        "diagnoses": [],
        "forecasts": [],
        "risk_summary": "",
        "rag_responses": [],
        "report": {},
        "report_md": "",
        "summary": "",
        "errors": [],
    }

    state = load_all_outputs(state)
    state = build_report(state)
    state = generate_markdown_report(state)
    state = generate_executive_summary(state)
    state = save_report(state)
    state = generate_summary(state)

    print("\n=== Report Agent Complete ===")
    print(f"Summary: {state['summary']}")
    print(f"\nExecutive Summary:\n{state['report'].get('executive_summary', 'N/A')}")
