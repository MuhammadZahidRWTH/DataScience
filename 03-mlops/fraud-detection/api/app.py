import os
import time
import json
import logging
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

# feature order matters — must match exactly what the pipeline was trained on
# if you retrain with different features update this list too
FEATURES = [
    "transaction_amount", "merchant_category", "hour_of_day",
    "day_of_week", "distance_from_home", "velocity_24h",
    "velocity_7d", "card_present", "online_transaction",
    "international", "num_transactions_24h", "avg_amount_7d",
    "std_amount_7d", "account_age_days", "credit_limit",
    "utilization_rate", "failed_attempts_24h", "device_change",
    "ip_change", "new_merchant",
]

# risk thresholds — somewhat arbitrary right now
# ideally calibrated against a cost matrix (false negative >> false positive in fraud)
# but 0.5 as the fraud cutoff works fine for this version
THRESHOLDS = {
    "HIGH":   0.8,
    "MEDIUM": 0.5,
    "LOW":    0.2,
}


# ── request / response schemas ────────────────────────────────────────────────

class Transaction(BaseModel):
    transaction_amount:   float = Field(..., description="transaction amount in EUR")
    merchant_category:    float
    hour_of_day:          float = Field(..., ge=0, le=23)
    day_of_week:          float = Field(..., ge=0, le=6)
    distance_from_home:   float = Field(..., ge=0)
    velocity_24h:         float = Field(..., ge=0)
    velocity_7d:          float = Field(..., ge=0)
    card_present:         float = Field(..., ge=0, le=1)
    online_transaction:   float = Field(..., ge=0, le=1)
    international:        float = Field(..., ge=0, le=1)
    num_transactions_24h: float = Field(..., ge=0)
    avg_amount_7d:        float = Field(..., ge=0)
    std_amount_7d:        float = Field(..., ge=0)
    account_age_days:     float = Field(..., ge=0)
    credit_limit:         float = Field(..., ge=0)
    utilization_rate:     float = Field(..., ge=0, le=1)
    failed_attempts_24h:  float = Field(..., ge=0)
    device_change:        float = Field(..., ge=0, le=1)
    ip_change:            float = Field(..., ge=0, le=1)
    new_merchant:         float = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_amount": 250.0, "merchant_category": 0.3,
                "hour_of_day": 2.0, "day_of_week": 5.0,
                "distance_from_home": 150.0, "velocity_24h": 8.0,
                "velocity_7d": 15.0, "card_present": 0.0,
                "online_transaction": 1.0, "international": 1.0,
                "num_transactions_24h": 8.0, "avg_amount_7d": 120.0,
                "std_amount_7d": 80.0, "account_age_days": 30.0,
                "credit_limit": 5000.0, "utilization_rate": 0.85,
                "failed_attempts_24h": 3.0, "device_change": 1.0,
                "ip_change": 1.0, "new_merchant": 1.0,
            }
        }
    }


class BatchRequest(BaseModel):
    transactions: List[Transaction]


class PredictionResponse(BaseModel):
    is_fraud:    bool
    fraud_score: float
    risk_level:  str
    latency_ms:  float


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total:       int
    fraud_count: int
    latency_ms:  float


# ── app state ─────────────────────────────────────────────────────────────────

# keeping state in a dict rather than global variables
# not ideal for multi-worker deployments but fine for now
# TODO: move to redis if we ever scale beyond a single instance
app_state = {
    "pipeline":      None,
    "metadata":      {},
    "request_count": 0,
    "fraud_count":   0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "model/pipeline.pkl")
    meta_path  = os.getenv("META_PATH",  "model/metadata.json")

    try:
        app_state["pipeline"] = joblib.load(model_path)
        logger.info(f"model loaded from {model_path}")
    except FileNotFoundError:
        # api still starts without a model — /health will reflect this
        # run model/train.py first to generate the pipeline
        logger.warning(f"model not found at {model_path} — run model/train.py first")

    try:
        with open(meta_path) as f:
            app_state["metadata"] = json.load(f)
    except FileNotFoundError:
        logger.warning("metadata.json not found — model info endpoint will be empty")

    yield


app = FastAPI(
    title="Fraud Detection API",
    description="real-time transaction fraud scoring",
    version="1.0.0",
    lifespan=lifespan,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def score_to_risk(score: float) -> str:
    if score >= THRESHOLDS["HIGH"]:
        return "HIGH"
    if score >= THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    if score >= THRESHOLDS["LOW"]:
        return "LOW"
    return "MINIMAL"


def run_inference(transaction: Transaction):
    if app_state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="model not loaded — run train.py first")
    df = pd.DataFrame([transaction.model_dump()])[FEATURES]
    t0 = time.time()
    proba   = app_state["pipeline"].predict_proba(df)[0][1]
    latency = (time.time() - t0) * 1000
    return float(proba), latency


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "healthy" if app_state["pipeline"] else "model_not_loaded",
        "model_loaded": app_state["pipeline"] is not None,
        "requests":     app_state["request_count"],
    }


@app.get("/model/info")
def model_info():
    return {
        "model":    "XGBoost + StandardScaler",
        "version":  "1.0.0",
        "metadata": app_state["metadata"],
        "features": FEATURES,
    }


@app.get("/metrics")
def metrics():
    total = app_state["request_count"]
    fraud = app_state["fraud_count"]
    return {
        "total_requests": total,
        "fraud_flagged":  fraud,
        "fraud_rate":     round(fraud / total, 4) if total > 0 else 0.0,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    proba, latency = run_inference(transaction)
    is_fraud = proba >= THRESHOLDS["MEDIUM"]

    app_state["request_count"] += 1
    if is_fraud:
        app_state["fraud_count"] += 1

    logger.info(f"score={proba:.4f} risk={score_to_risk(proba)} latency={latency:.1f}ms")

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_score=round(proba, 4),
        risk_level=score_to_risk(proba),
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if app_state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="model not loaded — run train.py first")

    t0   = time.time()
    rows = [t.model_dump() for t in request.transactions]
    df   = pd.DataFrame(rows)[FEATURES]

    probas  = app_state["pipeline"].predict_proba(df)[:, 1]
    latency = (time.time() - t0) * 1000

    predictions = [
        PredictionResponse(
            is_fraud=bool(p >= THRESHOLDS["MEDIUM"]),
            fraud_score=round(float(p), 4),
            risk_level=score_to_risk(p),
            latency_ms=round(latency / len(probas), 2),
        )
        for p in probas
    ]

    fraud_count = sum(1 for p in predictions if p.is_fraud)
    app_state["request_count"] += len(predictions)
    app_state["fraud_count"]   += fraud_count

    return BatchResponse(
        predictions=predictions,
        total=len(predictions),
        fraud_count=fraud_count,
        latency_ms=round(latency, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)