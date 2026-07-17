# dashboard/app.py
# MaintenanceGPT Streamlit Dashboard

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_PROCESSED

st.set_page_config(
    page_title="MaintenanceGPT",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card { background: #1e2435; border-radius: 10px; padding: 1rem; border-left: 4px solid #c8963a; }
.high-priority { border-left-color: #dc3545; }
.medium-priority { border-left-color: #ffc107; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_anomalies():
    try:
        return pd.read_csv(DATA_PROCESSED / "anomalies.csv", parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_diagnoses():
    try:
        with open(DATA_PROCESSED / "diagnoses.json", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return []

@st.cache_data
def load_forecasts():
    try:
        with open(DATA_PROCESSED / "forecasts.json", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def load_report():
    try:
        with open(DATA_PROCESSED / "maintenance_report.json", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data
def load_procedures():
    try:
        with open(DATA_PROCESSED / "rag_responses.json", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return []

@st.cache_data
def load_sensor():
    try:
        return pd.read_csv(DATA_PROCESSED / "sensor_clean.csv", parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()


# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/MaintenanceGPT-Sovereign_AI-gold?style=for-the-badge")
    st.markdown("### MaintenanceGPT")
    st.markdown("**Sovereign Agentic AI**  \nIndustrial Predictive Maintenance")
    st.markdown("---")
    st.markdown("**Pipeline:**")
    st.markdown("✓ Ingestion Agent")
    st.markdown("✓ Diagnosis Agent")
    st.markdown("✓ Forecasting Agent")
    st.markdown("✓ RAG Knowledge Agent")
    st.markdown("✓ Report Agent")
    st.markdown("---")
    st.markdown("Runs fully locally · Ollama · No cloud · GDPR compliant")
    st.markdown("---")
    st.markdown("*Sovereign Agentic AI · Made in Aachen, Germany*")
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# Load data
anomalies_df = load_anomalies()
diagnoses = load_diagnoses()
forecasts_data = load_forecasts()
report = load_report()
procedures = load_procedures()
sensor_df = load_sensor()

forecasts = forecasts_data.get("forecasts", [])
risk_summary = forecasts_data.get("risk_summary", "")

# Header
st.title("MaintenanceGPT — Maintenance Intelligence Dashboard")
st.markdown("**Sovereign multi-agent AI for industrial predictive maintenance** — runs fully locally via Ollama")
st.markdown("---")

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)
total_anomalies = len(anomalies_df) if not anomalies_df.empty else 0
fault_diagnoses = [d for d in diagnoses if d.get("predicted_fault") not in ("none", "unknown")]
bearing_wear = sum(1 for d in diagnoses if d.get("predicted_fault") == "bearing_wear")
overheating = sum(1 for d in diagnoses if d.get("predicted_fault") == "overheating")
high_priority = report.get("high_priority_count", 0)

with col1:
    st.metric("Total Anomalies", total_anomalies)
with col2:
    st.metric("Faults Detected", len(fault_diagnoses))
with col3:
    st.metric("Bearing Wear", bearing_wear)
with col4:
    st.metric("Overheating", overheating)
with col5:
    st.metric("High Priority", high_priority)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Anomalies", "Diagnosis", "Forecasts", "Procedures", "Report"
])

# Tab 1: Anomalies
with tab1:
    st.subheader("Sensor Anomaly Detection")
    if not anomalies_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            machine_counts = anomalies_df["machine_id"].value_counts().reset_index()
            machine_counts.columns = ["Machine", "Anomalies"]
            fig = px.bar(machine_counts, x="Machine", y="Anomalies", title="Anomalies per Machine",
                         color="Anomalies", color_continuous_scale="Oranges")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sensor_counts = anomalies_df["sensor"].value_counts().reset_index()
            sensor_counts.columns = ["Sensor", "Anomalies"]
            fig2 = px.pie(sensor_counts, values="Anomalies", names="Sensor",
                          title="Anomalies by Sensor Type",
                          color_discrete_sequence=px.colors.sequential.Oranges_r)
            st.plotly_chart(fig2, use_container_width=True)
        if "z_score" in anomalies_df.columns:
            fig3 = px.histogram(anomalies_df, x="z_score", color="machine_id",
                                title="Z-Score Distribution of Anomalies", nbins=30, barmode="overlay")
            st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(anomalies_df[["timestamp", "machine_id", "sensor", "z_score"]].head(50),
                     use_container_width=True)
    else:
        st.warning("No anomaly data found. Run the pipeline first.")

# Tab 2: Diagnosis
with tab2:
    st.subheader("Hybrid AI Fault Diagnosis")
    st.markdown("*Isolation Forest (unsupervised) + PyTorch MLP (supervised) + LLM explanation*")
    if diagnoses:
        fault_counts = {}
        for d in diagnoses:
            ft = d.get("predicted_fault", "unknown")
            fault_counts[ft] = fault_counts.get(ft, 0) + 1
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=list(fault_counts.values()), names=list(fault_counts.keys()),
                         title="Fault Type Distribution",
                         color_discrete_map={"none": "#6c757d", "bearing_wear": "#c8963a", "overheating": "#dc3545"})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fault_df = pd.DataFrame([d for d in diagnoses if d.get("predicted_fault") not in ("none", "unknown")])
            if not fault_df.empty and "confidence" in fault_df.columns:
                fig2 = px.box(fault_df, x="predicted_fault", y="confidence",
                              title="Model Confidence by Fault Type", color="predicted_fault",
                              color_discrete_map={"bearing_wear": "#c8963a", "overheating": "#dc3545"})
                st.plotly_chart(fig2, use_container_width=True)
        st.subheader("LLM Root Cause Explanations")
        explained = [d for d in diagnoses if "llm_explanation" in d]
        for d in explained:
            with st.expander(f"[HIGH] {d['machine_id']} — {d['predicted_fault']} (confidence: {d.get('confidence', 'N/A')})"):
                st.markdown(f"**Sensor:** {d['sensor']}")
                st.markdown(f"**Z-Score:** {d.get('z_score', 'N/A')}")
                st.markdown(f"**Isolation Score:** {d.get('isolation_score', 'N/A')}")
                st.markdown(f"**Root Cause:** {d['llm_explanation']}")
    else:
        st.warning("No diagnosis data found. Run the pipeline first.")

# Tab 3: Forecasts
with tab3:
    st.subheader("48h Sensor Forecasts (Prophet)")
    st.info(f"**Risk Assessment:** {risk_summary}")
    if forecasts:
        forecast_df = pd.DataFrame([f for f in forecasts if "error" not in f])
        if not forecast_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(forecast_df, x="sensor", y="failure_probability", color="machine_id",
                             title="48h Failure Probability by Sensor", barmode="group")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = go.Figure()
                for _, row in forecast_df.iterrows():
                    fig2.add_trace(go.Bar(
                        name=f"{row['machine_id']} — {row['sensor']}",
                        x=["Current Mean", "Forecast Mean 48h", "Danger Threshold"],
                        y=[row["current_mean"], row["forecast_mean_48h"], row["danger_threshold"]],
                    ))
                fig2.update_layout(title="Current vs Forecast vs Threshold", barmode="group", showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(forecast_df, use_container_width=True)
    else:
        st.warning("No forecast data found. Run the pipeline first.")

# Tab 4: Procedures
with tab4:
    st.subheader("RAG Knowledge Base — Maintenance Procedures")
    st.markdown("*Procedures retrieved from machine manuals via FAISS vector search + Ollama*")
    if procedures:
        for p in procedures:
            with st.expander(f"{p['machine_id']} — {p['fault']} ({p['sensor']})"):
                st.markdown("**Recommended Procedure:**")
                st.markdown(p["manual_procedure"])
                if p.get("sources"):
                    st.markdown("**Manual excerpts used:**")
                    for src in p["sources"]:
                        st.code(src, language=None)
    else:
        st.warning("No procedures found. Run the pipeline first.")

# Tab 5: Report
with tab5:
    st.subheader("Maintenance Intelligence Report")
    if report:
        st.markdown(f"**Report ID:** `{report.get('report_id', 'N/A')}`")
        st.markdown(f"**Generated:** {report.get('generated_at', 'N/A')}")
        st.markdown(f"**System:** {report.get('generated_by', 'N/A')}")
        st.markdown("---")
        st.markdown("### Executive Summary")
        st.info(report.get("executive_summary", "N/A"))
        st.markdown("### Action Items")
        for item in report.get("action_items", []):
            priority_label = "[HIGH]" if item["priority"] == "HIGH" else "[MEDIUM]"
            with st.expander(f"{priority_label} {item['machine']} — {item['fault']}"):
                st.markdown(f"**Sensor:** {item['sensor']}")
                st.markdown(f"**Confidence:** {item.get('confidence', 'N/A')}")
                st.markdown(f"**Diagnosis:** {item['diagnosis']}")
                st.markdown(f"**Procedure:** {item['procedure']}")
        try:
            with open(DATA_PROCESSED / "maintenance_report.md", encoding="utf-8") as f:
                report_md = f.read()
            st.download_button(
                label="Download Full Report (Markdown)",
                data=report_md,
                file_name=f"maintenance_report_{report.get('report_id', 'latest')}.md",
                mime="text/markdown",
            )
        except Exception:
            pass
    else:
        st.warning("No report found. Run the pipeline first.")