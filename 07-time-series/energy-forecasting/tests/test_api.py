import numpy as np
import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, 'src')

from api import app
from dataset import INPUT_WINDOW, FORECAST_HORIZON, FEATURES

client = TestClient(app)


def make_dummy_input():
    return [[float(i % 10)] * len(FEATURES) for i in range(INPUT_WINDOW)]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_info():
    r = client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert data["input_window_hours"] == INPUT_WINDOW
    assert data["forecast_horizon_hours"] == FORECAST_HORIZON
    assert len(data["features"]) == len(FEATURES)


def test_forecast_valid():
    r = client.post("/forecast", json={"features": make_dummy_input()})
    assert r.status_code == 200
    data = r.json()
    assert len(data["forecast_eur_mwh"]) == FORECAST_HORIZON
    assert data["forecast_hours"] == FORECAST_HORIZON
    assert data["latency_ms"] > 0


def test_forecast_wrong_length():
    r = client.post("/forecast", json={"features": [[1.0] * len(FEATURES)] * 10})
    assert r.status_code == 422


def test_anomaly_detection():
    features = make_dummy_input()
    actual = [100.0] * FORECAST_HORIZON
    r = client.post("/anomaly", json={"features": features, "actual": actual})
    assert r.status_code == 200
    data = r.json()
    assert "is_anomaly" in data
    assert "reconstruction_error" in data
    assert "threshold" in data
    assert data["confidence"] in ["low", "medium", "high"]


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code in [200, 404]
