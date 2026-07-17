# agents/ingestion/ingestion_agent.py
# Ingestion Agent — first agent in the MaintenanceGPT pipeline
#
# Responsibilities:
#   1. Load raw sensor data and maintenance logs
#   2. Clean and validate data quality
#   3. Detect anomalies using z-score on sensor readings
#   4. Output structured, processed data for downstream agents
#
# Design rationale:
#   - Implements LLM-based log cleaning (see: "Cleaning Maintenance Logs
#     with LLM Agents for Improved Predictive Maintenance", 2025)
#   - Sovereign AI: uses Ollama locally, no data leaves the machine

import pandas as pd
import numpy as np
from pathlib import Path
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import sys

# add project root to path so config.py is found from any subdirectory
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    DATA_RAW,
    DATA_PROCESSED,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    ANOMALY_THRESHOLD,
    WINDOW_SIZE,
)


# ── Agent State ────────────────────────────────────────────
class IngestionState(TypedDict):
    sensor_df: pd.DataFrame
    logs_df: pd.DataFrame
    anomalies_df: pd.DataFrame
    cleaned_logs: list[dict]
    summary: str
    errors: list[str]


# ── Data Loader ────────────────────────────────────────────
def load_data(state: IngestionState) -> IngestionState:
    """Load raw CSVs from data/raw/"""
    errors = []

    try:
        sensor_path = DATA_RAW / "sensor_readings.csv"
        sensor_df = pd.read_csv(sensor_path, parse_dates=["timestamp"])
        print(f"  [Ingestion] Loaded sensor data: {len(sensor_df)} rows")
    except Exception as e:
        sensor_df = pd.DataFrame()
        errors.append(f"Sensor load failed: {e}")

    try:
        logs_path = DATA_RAW / "maintenance_logs.csv"
        logs_df = pd.read_csv(logs_path, parse_dates=["timestamp"])
        print(f"  [Ingestion] Loaded maintenance logs: {len(logs_df)} rows")
    except Exception as e:
        logs_df = pd.DataFrame()
        errors.append(f"Logs load failed: {e}")

    return {**state, "sensor_df": sensor_df, "logs_df": logs_df, "errors": errors}


# ── Data Cleaner ───────────────────────────────────────────
def clean_sensor_data(state: IngestionState) -> IngestionState:
    """
    Clean sensor data:
    - Drop nulls
    - Clip physically impossible values
    - Add rolling statistics for downstream forecasting
    """
    df = state["sensor_df"].copy()

    if df.empty:
        return state

    # drop rows with any null sensor reading
    before = len(df)
    df = df.dropna(subset=["temperature_c", "vibration_mm_s", "pressure_bar"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  [Ingestion] Dropped {dropped} rows with null sensor values")

    # clip physically impossible values per sensor type
    df["temperature_c"] = df["temperature_c"].clip(lower=0, upper=200)
    df["vibration_mm_s"] = df["vibration_mm_s"].clip(lower=0, upper=20)
    df["pressure_bar"] = df["pressure_bar"].clip(lower=0, upper=50)

    # rolling mean and std per machine (used by forecasting agent)
    for col in ["temperature_c", "vibration_mm_s", "pressure_bar"]:
        df[f"{col}_rolling_mean"] = (
            df.groupby("machine_id")[col]
            .transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).mean())
        )
        df[f"{col}_rolling_std"] = (
            df.groupby("machine_id")[col]
            .transform(lambda x: x.rolling(WINDOW_SIZE, min_periods=1).std().fillna(0))
        )

    print(f"  [Ingestion] Sensor data cleaned — {len(df)} rows retained")
    return {**state, "sensor_df": df}


# ── Anomaly Detector ───────────────────────────────────────
def detect_anomalies(state: IngestionState) -> IngestionState:
    """
    Z-score anomaly detection on sensor readings.
    Flags readings > ANOMALY_THRESHOLD standard deviations from mean.
    """
    df = state["sensor_df"].copy()

    if df.empty:
        return {**state, "anomalies_df": pd.DataFrame()}

    anomaly_records = []

    for machine in df["machine_id"].unique():
        mdf = df[df["machine_id"] == machine].copy()

        for col in ["temperature_c", "vibration_mm_s", "pressure_bar"]:
            mean = mdf[col].mean()
            std = mdf[col].std()

            if std == 0:
                continue

            z_scores = (mdf[col] - mean) / std
            anomalies = mdf[np.abs(z_scores) > ANOMALY_THRESHOLD].copy()
            anomalies["sensor"] = col
            anomalies["z_score"] = z_scores[anomalies.index].round(3)
            anomaly_records.append(anomalies)

    if anomaly_records:
        anomalies_df = pd.concat(anomaly_records).drop_duplicates()
        anomalies_df = anomalies_df.sort_values("timestamp")
    else:
        anomalies_df = pd.DataFrame()

    print(f"  [Ingestion] Anomalies detected: {len(anomalies_df)}")
    return {**state, "anomalies_df": anomalies_df}


# ── LLM Log Cleaner ────────────────────────────────────────
def clean_logs_with_llm(state: IngestionState) -> IngestionState:
    """
    Use Ollama LLM to clean and structure messy maintenance log entries.
    Extracts: fault_category, severity, action_taken, follow_up_needed.

    This implements the approach from:
    'Cleaning Maintenance Logs with LLM Agents for Improved
    Predictive Maintenance' (2025)
    """
    logs_df = state["logs_df"].copy()

    if logs_df.empty:
        return {**state, "cleaned_logs": []}

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,  # deterministic for data cleaning
    )

    cleaned_logs = []

    # process first 20 logs to keep Day 1 fast
    # full pipeline processes all logs
    sample = logs_df.head(20)

    print(f"  [Ingestion] Cleaning {len(sample)} log entries with LLM...")

    for _, row in sample.iterrows():
        prompt = f"""You are an industrial maintenance data analyst.
Extract structured information from this maintenance log entry.

Log: "{row['description']}"
Machine: {row['machine_id']}
Resolved: {row['resolved']}

Respond in exactly this format (no extra text):
fault_category: <bearing_wear|overheating|pressure_issue|lubrication|scheduled|other>
severity: <low|medium|high|critical>
action_taken: <one sentence>
follow_up_needed: <yes|no>"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            parsed = _parse_llm_response(response.content)
            cleaned_logs.append({
                "log_id": row["log_id"],
                "machine_id": row["machine_id"],
                "timestamp": str(row["timestamp"]),
                "original_description": row["description"],
                **parsed,
            })
        except Exception as e:
            # don't fail pipeline on single log error
            cleaned_logs.append({
                "log_id": row["log_id"],
                "machine_id": row["machine_id"],
                "timestamp": str(row["timestamp"]),
                "original_description": row["description"],
                "fault_category": "unknown",
                "severity": "unknown",
                "action_taken": "parse_error",
                "follow_up_needed": "unknown",
                "error": str(e),
            })

    print(f"  [Ingestion] LLM cleaning complete — {len(cleaned_logs)} logs processed")
    return {**state, "cleaned_logs": cleaned_logs}


def _parse_llm_response(text: str) -> dict:
    """Parse structured LLM response into dict."""
    result = {
        "fault_category": "unknown",
        "severity": "unknown",
        "action_taken": "unknown",
        "follow_up_needed": "unknown",
    }
    for line in text.strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if key in result:
                result[key] = value
    return result


# ── Summary Generator ──────────────────────────────────────
def generate_summary(state: IngestionState) -> IngestionState:
    """Generate ingestion summary passed to next agent."""
    n_sensor = len(state["sensor_df"])
    n_anomalies = len(state["anomalies_df"]) if not state["anomalies_df"].empty else 0
    n_logs = len(state["cleaned_logs"])
    machines = state["sensor_df"]["machine_id"].unique().tolist() if not state["sensor_df"].empty else []

    summary = (
        f"Ingestion complete. "
        f"Sensor records: {n_sensor}. "
        f"Anomalies flagged: {n_anomalies}. "
        f"Log entries cleaned: {n_logs}. "
        f"Machines monitored: {', '.join(machines)}."
    )
    print(f"  [Ingestion] {summary}")
    return {**state, "summary": summary}


# ── Save Processed Data ────────────────────────────────────
def save_processed(state: IngestionState) -> IngestionState:
    """Persist cleaned data to data/processed/ for downstream agents."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if not state["sensor_df"].empty:
        state["sensor_df"].to_csv(
            DATA_PROCESSED / "sensor_clean.csv", index=False
        )

    if not state["anomalies_df"].empty:
        state["anomalies_df"].to_csv(
            DATA_PROCESSED / "anomalies.csv", index=False
        )

    import json
    with open(DATA_PROCESSED / "cleaned_logs.json", "w") as f:
        json.dump(state["cleaned_logs"], f, indent=2)

    print(f"  [Ingestion] Processed data saved to data/processed/")
    return state


# ── Run standalone ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== MaintenanceGPT — Ingestion Agent ===\n")

    state: IngestionState = {
        "sensor_df": pd.DataFrame(),
        "logs_df": pd.DataFrame(),
        "anomalies_df": pd.DataFrame(),
        "cleaned_logs": [],
        "summary": "",
        "errors": [],
    }

    state = load_data(state)
    state = clean_sensor_data(state)
    state = detect_anomalies(state)
    state = clean_logs_with_llm(state)
    state = generate_summary(state)
    state = save_processed(state)

    print("\n=== Ingestion Agent Complete ===")
    print(f"Summary: {state['summary']}")
    if state["errors"]:
        print(f"Errors: {state['errors']}")
