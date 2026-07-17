# api/main.py
# MaintenanceGPT FastAPI Service
#
# Exposes the full agentic pipeline via REST endpoints.
# Designed for production deployment with Docker.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_PROCESSED

app = FastAPI(
    title="MaintenanceGPT API",
    description="Sovereign Agentic AI for Industrial Predictive Maintenance",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "MaintenanceGPT",
        "timestamp": datetime.now().isoformat(),
        "sovereign": True,
        "cloud_dependency": False,
    }

@app.get("/status")
def get_status():
    files = {
        "sensor_clean": (DATA_PROCESSED / "sensor_clean.csv").exists(),
        "anomalies": (DATA_PROCESSED / "anomalies.csv").exists(),
        "diagnoses": (DATA_PROCESSED / "diagnoses.json").exists(),
        "forecasts": (DATA_PROCESSED / "forecasts.json").exists(),
        "rag_responses": (DATA_PROCESSED / "rag_responses.json").exists(),
        "report": (DATA_PROCESSED / "maintenance_report.json").exists(),
    }
    return {
        "pipeline_complete": all(files.values()),
        "outputs": files,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/report")
def get_report():
    try:
        with open(DATA_PROCESSED / "maintenance_report.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "No report found. Run the pipeline first."}

@app.get("/diagnoses")
def get_diagnoses(limit: int = 20):
    try:
        with open(DATA_PROCESSED / "diagnoses.json") as f:
            diagnoses = json.load(f)
        faults = [d for d in diagnoses if d.get("predicted_fault") not in ("none", "unknown")]
        return {
            "total": len(diagnoses),
            "faults_detected": len(faults),
            "diagnoses": faults[:limit],
        }
    except FileNotFoundError:
        return {"error": "No diagnoses found. Run the pipeline first."}

@app.get("/forecasts")
def get_forecasts():
    try:
        with open(DATA_PROCESSED / "forecasts.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "No forecasts found. Run the pipeline first."}

@app.get("/procedures")
def get_procedures():
    try:
        with open(DATA_PROCESSED / "rag_responses.json") as f:
            return {"procedures": json.load(f)}
    except FileNotFoundError:
        return {"error": "No procedures found. Run the pipeline first."}

@app.get("/anomalies")
def get_anomalies(machine_id: str = None, limit: int = 50):
    try:
        df = pd.read_csv(DATA_PROCESSED / "anomalies.csv")
        if machine_id:
            df = df[df["machine_id"] == machine_id]
        return {
            "total": len(df),
            "anomalies": df.head(limit).to_dict(orient="records")
        }
    except FileNotFoundError:
        return {"error": "No anomalies found. Run the pipeline first."}