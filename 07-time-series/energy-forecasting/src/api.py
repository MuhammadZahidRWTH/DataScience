import json
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib

from dataset import load_and_split, fit_scaler, FEATURES, INPUT_WINDOW, FORECAST_HORIZON
from model import TCN


# ── State ─────────────────────────────────────────────────────────────────────

class AppState:
    model: TCN = None
    scaler = None
    anomaly_threshold: float = None
    baseline_results: dict = None

state = AppState()


def load_artifacts():
    model_path = Path("models/best_tcn.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    train_df, _, _ = load_and_split()
    state.scaler = fit_scaler(train_df)

    state.model = TCN(n_features=len(FEATURES))
    state.model.load_state_dict(torch.load(model_path, map_location="cpu"))
    state.model.eval()

    anomaly_path = Path("data/anomaly_results.json")
    if anomaly_path.exists():
        with open(anomaly_path) as f:
            res = json.load(f)
        state.anomaly_threshold = res.get("threshold", 0.63)

    baseline_path = Path("data/baseline_results.json")
    if baseline_path.exists():
        with open(baseline_path) as f:
            state.baseline_results = json.load(f)

    print(f"Model loaded. Threshold: {state.anomaly_threshold:.4f}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(
    title="Energy Price Forecasting API",
    description="PyTorch TCN for European electricity price forecasting and anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description=f"Input window of shape [{INPUT_WINDOW}, {len(FEATURES)}]. "
                    f"Features: {FEATURES}"
    )

class ForecastResponse(BaseModel):
    forecast_eur_mwh: list[float]
    forecast_hours: int
    latency_ms: float
    model: str = "PyTorch TCN"

class AnomalyRequest(BaseModel):
    features: list[list[float]] = Field(
        ...,
        description="Same format as ForecastRequest"
    )
    actual: list[float] = Field(
        ...,
        description=f"Actual prices for the next {FORECAST_HORIZON} hours"
    )

class AnomalyResponse(BaseModel):
    is_anomaly: bool
    reconstruction_error: float
    threshold: float
    confidence: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "threshold": state.anomaly_threshold,
    }


@app.get("/info")
def info():
    return {
        "model": "Temporal Convolutional Network (TCN)",
        "input_window_hours": INPUT_WINDOW,
        "forecast_horizon_hours": FORECAST_HORIZON,
        "features": FEATURES,
        "baselines": state.baseline_results,
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if len(req.features) != INPUT_WINDOW:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {INPUT_WINDOW} timesteps, got {len(req.features)}"
        )
    if any(len(row) != len(FEATURES) for row in req.features):
        raise HTTPException(
            status_code=422,
            detail=f"Each row must have {len(FEATURES)} features: {FEATURES}"
        )

    x_raw = np.array(req.features, dtype=np.float32)
    x_scaled = state.scaler.transform(x_raw)
    x_tensor = torch.tensor(x_scaled).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        pred_scaled = state.model(x_tensor).squeeze(0).numpy()
    latency_ms = (time.perf_counter() - t0) * 1000

    target_idx = FEATURES.index("price_eur_mwh")
    scale = state.scaler.scale_[target_idx]
    mean  = state.scaler.mean_[target_idx]
    pred_real = (pred_scaled * scale + mean).tolist()

    return ForecastResponse(
        forecast_eur_mwh=pred_real,
        forecast_hours=FORECAST_HORIZON,
        latency_ms=round(latency_ms, 3),
    )


@app.post("/anomaly", response_model=AnomalyResponse)
def detect_anomaly(req: AnomalyRequest):
    if state.anomaly_threshold is None:
        raise HTTPException(status_code=503, detail="Anomaly threshold not loaded")

    forecast_resp = forecast(ForecastRequest(features=req.features))
    pred = np.array(forecast_resp.forecast_eur_mwh)
    actual = np.array(req.actual)

    if len(actual) != FORECAST_HORIZON:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {FORECAST_HORIZON} actual values"
        )

    error = float(np.mean(np.abs(pred - actual)))
    is_anomaly = error > state.anomaly_threshold

    if error > state.anomaly_threshold * 1.5:
        confidence = "high"
    elif error > state.anomaly_threshold:
        confidence = "medium"
    else:
        confidence = "low"

    return AnomalyResponse(
        is_anomaly=is_anomaly,
        reconstruction_error=round(error, 6),
        threshold=round(state.anomaly_threshold, 6),
        confidence=confidence,
    )


@app.get("/metrics")
def metrics():
    tcn_path = Path("data/tcn_results.json")
    if not tcn_path.exists():
        raise HTTPException(status_code=404, detail="TCN results not found")
    with open(tcn_path) as f:
        tcn = json.load(f)
    return {
        "tcn": tcn,
        "baselines": state.baseline_results,
    }
