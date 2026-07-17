# agents/diagnosis/diagnosis_agent.py
# Diagnosis Agent — second agent in the MaintenanceGPT pipeline
#
# Responsibilities:
#   1. Load anomalies and cleaned logs from ingestion agent
#   2. Score anomalies with Isolation Forest (unsupervised)
#   3. Classify fault type with PyTorch MLP (supervised)
#   4. Use Ollama LLM for root cause explanation
#
# Design rationale:
#   - Hybrid AI: Isolation Forest + PyTorch + LLM reasoning
#   - Unsupervised + supervised combination mirrors real
#     industrial deployments where labels are scarce
#   - Explainable outputs for operator trust

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    DATA_PROCESSED,
    DATA_RAW,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)


# ── Agent State ────────────────────────────────────────────
class DiagnosisState(TypedDict):
    anomalies_df: pd.DataFrame
    cleaned_logs: list[dict]
    diagnoses: list[dict]
    model_report: str
    summary: str
    errors: list[str]


# ── PyTorch Fault Classifier ───────────────────────────────
class FaultClassifierMLP(nn.Module):
    """
    Small MLP for fault type classification from sensor features.
    Deliberately lightweight — designed for edge deployment
    #   Suitable for edge deployment on industrial hardware.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Load Ingestion Output ──────────────────────────────────
def load_ingestion_output(state: DiagnosisState) -> DiagnosisState:
    """Load processed data produced by ingestion agent."""
    errors = []

    try:
        anomalies_df = pd.read_csv(
            DATA_PROCESSED / "anomalies.csv",
            parse_dates=["timestamp"]
        )
        print(f"  [Diagnosis] Loaded anomalies: {len(anomalies_df)} rows")
    except Exception as e:
        anomalies_df = pd.DataFrame()
        errors.append(f"Anomalies load failed: {e}")

    try:
        with open(DATA_PROCESSED / "cleaned_logs.json") as f:
            cleaned_logs = json.load(f)
        print(f"  [Diagnosis] Loaded cleaned logs: {len(cleaned_logs)} entries")
    except Exception as e:
        cleaned_logs = []
        errors.append(f"Logs load failed: {e}")

    return {
        **state,
        "anomalies_df": anomalies_df,
        "cleaned_logs": cleaned_logs,
        "errors": errors
    }


# ── Isolation Forest Scoring ───────────────────────────────
def score_with_isolation_forest(state: DiagnosisState) -> DiagnosisState:
    """
    Unsupervised anomaly scoring using Isolation Forest.
    Assigns an anomaly score to each detected anomaly.
    Score < 0 = anomalous, closer to -1 = more anomalous.

    Why Isolation Forest here:
    - No labels required — critical for real industrial data
      where fault labels are sparse or missing
    - Fast inference — suitable for edge deployment
    #   Widely used in industrial fault detection applications.
    """
    anomalies_df = state["anomalies_df"].copy()

    if anomalies_df.empty:
        return state

    features = ["temperature_c", "vibration_mm_s", "pressure_bar"]
    available = [f for f in features if f in anomalies_df.columns]

    X = anomalies_df[available].fillna(0).values

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.1,   # ~10% of data expected to be anomalous
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X)

    anomalies_df["isolation_score"] = iso_forest.score_samples(X).round(4)
    anomalies_df["iso_label"] = iso_forest.predict(X)  # -1=anomaly, 1=normal

    confirmed = (anomalies_df["iso_label"] == -1).sum()
    print(f"  [Diagnosis] Isolation Forest — confirmed anomalies: {confirmed}/{len(anomalies_df)}")

    return {**state, "anomalies_df": anomalies_df}


# ── Train PyTorch MLP ──────────────────────────────────────
def train_pytorch_classifier(state: DiagnosisState) -> DiagnosisState:
    """
    Train lightweight PyTorch MLP for fault type classification.
    Uses full sensor dataset with known fault labels.

    Hybrid AI approach:
    - Isolation Forest handles the unsupervised scoring
    - PyTorch MLP handles the supervised classification
    - Together they handle both labeled and unlabeled scenarios
    """
    try:
        sensor_df = pd.read_csv(
            DATA_RAW / "sensor_readings.csv",
            parse_dates=["timestamp"]
        )

        # balance dataset: fault rows + equal normal rows
        fault_df = sensor_df[sensor_df["fault_flag"] == 1].copy()
        normal_df = sensor_df[sensor_df["fault_flag"] == 0].sample(
            n=min(len(fault_df) * 3, len(sensor_df[sensor_df["fault_flag"] == 0])),
            random_state=42
        )
        train_df = pd.concat([fault_df, normal_df]).reset_index(drop=True)

        features = ["temperature_c", "vibration_mm_s", "pressure_bar", "rpm"]
        X = train_df[features].fillna(0).values
        y = train_df["fault_type"].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )

        # convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_test_t = torch.FloatTensor(X_test)
        y_test_t = torch.LongTensor(y_test)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # model
        num_classes = len(le.classes_)
        model = FaultClassifierMLP(
            input_dim=len(features),
            hidden_dim=64,
            num_classes=num_classes
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # train for 15 epochs — enough for this dataset size
        print(f"  [Diagnosis] Training PyTorch MLP — {num_classes} fault classes...")
        model.train()
        for epoch in range(15):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(loader)
                print(f"    Epoch {epoch+1}/15 — loss: {avg_loss:.4f}")

        # evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, predicted = torch.max(test_outputs, 1)
            report = classification_report(
                y_test_t.numpy(),
                predicted.numpy(),
                target_names=le.classes_
            )
        print(f"  [Diagnosis] PyTorch MLP evaluation:\n{report}")

        # store in state
        state["_model"] = model
        state["_encoder"] = le
        state["_scaler"] = scaler
        state["model_report"] = report

    except Exception as e:
        state["errors"].append(f"PyTorch training failed: {e}")
        print(f"  [Diagnosis] ERROR training PyTorch MLP: {e}")

    return state


# ── Classify Anomalies ─────────────────────────────────────
def classify_anomalies(state: DiagnosisState) -> DiagnosisState:
    """Run PyTorch MLP on confirmed anomalies from Isolation Forest."""
    anomalies_df = state["anomalies_df"].copy()

    if anomalies_df.empty or "_model" not in state:
        return {**state, "diagnoses": []}

    model = state["_model"]
    le = state["_encoder"]
    scaler = state["_scaler"]

    features = ["temperature_c", "vibration_mm_s", "pressure_bar", "rpm"]
    for f in features:
        if f not in anomalies_df.columns:
            anomalies_df[f] = 0
    anomalies_df["rpm"] = anomalies_df["rpm"].fillna(1450.0)

    X = anomalies_df[features].fillna(0).values
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    anomalies_df["predicted_fault"] = le.inverse_transform(predicted.numpy())
    anomalies_df["confidence"] = confidence.numpy().round(3)

    diagnoses = anomalies_df[[
        "timestamp", "machine_id", "sensor",
        "predicted_fault", "confidence",
        "z_score", "isolation_score"
    ]].to_dict(orient="records")

    print(f"  [Diagnosis] Classified {len(diagnoses)} anomalies with PyTorch MLP")
    return {**state, "diagnoses": diagnoses}


# ── LLM Root Cause Explanation ─────────────────────────────
def explain_with_llm(state: DiagnosisState) -> DiagnosisState:
    """
    Ollama LLM generates root cause explanation for top fault anomalies.
    Hybrid AI: Isolation Forest scores → PyTorch classifies → LLM explains.
    """
    diagnoses = state["diagnoses"]

    if not diagnoses:
        return state

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

    # only explain actual faults — skip "none" class
    fault_indices = [
        i for i, d in enumerate(diagnoses)
        if d.get("predicted_fault") not in ("none", "unknown")
    ]

    # sort by confidence, take top 5
    top_indices = sorted(
        fault_indices,
        key=lambda i: diagnoses[i]["confidence"],
        reverse=True
    )[:5]

    print(f"  [Diagnosis] Generating LLM explanations for top {len(top_indices)} fault diagnoses...")

    for idx in top_indices:
        diag = diagnoses[idx]

        prompt = f"""You are an industrial maintenance expert at a German manufacturing plant.

A sensor anomaly was detected and classified by a hybrid AI system
(Isolation Forest + PyTorch neural network).

Machine: {diag['machine_id']}
Sensor triggered: {diag['sensor']}
Z-score: {diag['z_score']}
Isolation score: {diag.get('isolation_score', 'N/A')}
Predicted fault: {diag['predicted_fault']}
Model confidence: {diag['confidence']}

In 2 sentences: explain the likely root cause and recommend
the most important immediate action. Be specific and technical."""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            diagnoses[idx]["llm_explanation"] = response.content.strip()
            print(f"    → {diag['machine_id']} | {diag['predicted_fault']} | explained")
        except Exception as e:
            diagnoses[idx]["llm_explanation"] = f"Explanation unavailable: {e}"

    return {**state, "diagnoses": diagnoses}


# ── Save Diagnoses ─────────────────────────────────────────
def save_diagnoses(state: DiagnosisState) -> DiagnosisState:
    """Save diagnosis results for downstream agents."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    diagnoses = state["diagnoses"]
    for d in diagnoses:
        if hasattr(d.get("timestamp"), "isoformat"):
            d["timestamp"] = d["timestamp"].isoformat()
        elif not isinstance(d.get("timestamp"), str):
            d["timestamp"] = str(d.get("timestamp", ""))

    with open(DATA_PROCESSED / "diagnoses.json", "w") as f:
        json.dump(diagnoses, f, indent=2, default=str)

    print(f"  [Diagnosis] Saved {len(diagnoses)} diagnoses to data/processed/")
    return state


# ── Summary ────────────────────────────────────────────────
def generate_summary(state: DiagnosisState) -> DiagnosisState:
    diagnoses = state["diagnoses"]
    if diagnoses:
        fault_counts = {}
        for d in diagnoses:
            ft = d.get("predicted_fault", "unknown")
            fault_counts[ft] = fault_counts.get(ft, 0) + 1
        fault_str = ", ".join(f"{k}: {v}" for k, v in fault_counts.items())
    else:
        fault_str = "none"

    summary = (
        f"Diagnosis complete. "
        f"Anomalies classified: {len(diagnoses)}. "
        f"Fault breakdown: {fault_str}."
    )
    print(f"  [Diagnosis] {summary}")
    return {**state, "summary": summary}


# ── Run standalone ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== MaintenanceGPT — Diagnosis Agent ===\n")

    state: DiagnosisState = {
        "anomalies_df": pd.DataFrame(),
        "cleaned_logs": [],
        "diagnoses": [],
        "model_report": "",
        "summary": "",
        "errors": [],
    }

    state = load_ingestion_output(state)
    state = score_with_isolation_forest(state)
    state = train_pytorch_classifier(state)
    state = classify_anomalies(state)
    state = explain_with_llm(state)
    state = save_diagnoses(state)
    state = generate_summary(state)

    print("\n=== Diagnosis Agent Complete ===")
    print(f"Summary: {state['summary']}")
    if state["errors"]:
        print(f"Errors: {state['errors']}")
