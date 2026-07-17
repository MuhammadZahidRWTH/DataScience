# tests/test_pipeline.py
# Basic tests for MaintenanceGPT pipeline components

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Config Tests ───────────────────────────────────────────
def test_config_loads():
    from config import (
        OLLAMA_MODEL, ANOMALY_THRESHOLD,
        FORECAST_HORIZON_HOURS, TOP_K_RETRIEVAL
    )
    assert OLLAMA_MODEL == "llama3.2"
    assert ANOMALY_THRESHOLD > 0
    assert FORECAST_HORIZON_HOURS == 48
    assert TOP_K_RETRIEVAL > 0


# ── Ingestion Tests ────────────────────────────────────────
def test_anomaly_detection():
    from agents.ingestion.ingestion_agent import detect_anomalies

    # create synthetic sensor data with known anomaly
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "machine_id": ["CNC_Mill_01"] * n,
        "temperature_c": [65.0] * n,
        "vibration_mm_s": [0.8] * n,
        "pressure_bar": [4.2] * n,
        "rpm": [1450.0] * n,
        "fault_flag": [0] * n,
        "fault_type": ["none"] * n,
    })
    # inject obvious anomaly
    df.loc[50, "temperature_c"] = 200.0

    state = {
        "sensor_df": df,
        "logs_df": pd.DataFrame(),
        "anomalies_df": pd.DataFrame(),
        "cleaned_logs": [],
        "summary": "",
        "errors": [],
    }

    result = detect_anomalies(state)
    assert not result["anomalies_df"].empty
    assert len(result["anomalies_df"]) >= 1


def test_clean_sensor_data():
    from agents.ingestion.ingestion_agent import clean_sensor_data

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
        "machine_id": ["CNC_Mill_01"] * 10,
        "temperature_c": [65.0, None, 65.0, 300.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0],
        "vibration_mm_s": [0.8] * 10,
        "pressure_bar": [4.2] * 10,
        "rpm": [1450.0] * 10,
        "fault_flag": [0] * 10,
        "fault_type": ["none"] * 10,
    })

    state = {
        "sensor_df": df,
        "logs_df": pd.DataFrame(),
        "anomalies_df": pd.DataFrame(),
        "cleaned_logs": [],
        "summary": "",
        "errors": [],
    }

    result = clean_sensor_data(state)
    # null row dropped
    assert len(result["sensor_df"]) == 9
    # temperature clipped to max 200
    assert result["sensor_df"]["temperature_c"].max() <= 200.0


# ── Diagnosis Tests ────────────────────────────────────────
def test_isolation_forest():
    from agents.diagnosis.diagnosis_agent import score_with_isolation_forest

    n = 50
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "machine_id": ["CNC_Mill_01"] * n,
        "temperature_c": [65.0] * n,
        "vibration_mm_s": [0.8] * n,
        "pressure_bar": [4.2] * n,
        "sensor": ["temperature_c"] * n,
        "z_score": [2.6] * n,
    })
    # inject anomaly
    df.loc[25, "temperature_c"] = 150.0

    state = {
        "anomalies_df": df,
        "cleaned_logs": [],
        "diagnoses": [],
        "model_report": "",
        "summary": "",
        "errors": [],
    }

    result = score_with_isolation_forest(state)
    assert "isolation_score" in result["anomalies_df"].columns
    assert "iso_label" in result["anomalies_df"].columns


# ── FastAPI Tests ──────────────────────────────────────────
def test_api_health():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["sovereign"] is True
    assert data["cloud_dependency"] is False


def test_api_status():
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_complete" in data
    assert "outputs" in data