import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# sample transactions for testing
LEGIT = {
    "transaction_amount": 45.0, "merchant_category": 0.2,
    "hour_of_day": 14.0, "day_of_week": 2.0,
    "distance_from_home": 2.0, "velocity_24h": 2.0,
    "velocity_7d": 8.0, "card_present": 1.0,
    "online_transaction": 0.0, "international": 0.0,
    "num_transactions_24h": 2.0, "avg_amount_7d": 50.0,
    "std_amount_7d": 10.0, "account_age_days": 365.0,
    "credit_limit": 5000.0, "utilization_rate": 0.2,
    "failed_attempts_24h": 0.0, "device_change": 0.0,
    "ip_change": 0.0, "new_merchant": 0.0,
}

FRAUD = {
    "transaction_amount": 4500.0, "merchant_category": 0.9,
    "hour_of_day": 3.0, "day_of_week": 6.0,
    "distance_from_home": 5000.0, "velocity_24h": 25.0,
    "velocity_7d": 50.0, "card_present": 0.0,
    "online_transaction": 1.0, "international": 1.0,
    "num_transactions_24h": 25.0, "avg_amount_7d": 60.0,
    "std_amount_7d": 200.0, "account_age_days": 5.0,
    "credit_limit": 1000.0, "utilization_rate": 0.99,
    "failed_attempts_24h": 5.0, "device_change": 1.0,
    "ip_change": 1.0, "new_merchant": 1.0,
}


# ── data tests ────────────────────────────────────────────────────────────────

class TestDataGeneration:

    def test_shape(self):
        import sys
        sys.path.insert(0, ".")
        from data.generate_data import generate_fraud_data
        df = generate_fraud_data(n_samples=1000)
        assert df.shape == (1000, 21)
        assert "is_fraud" in df.columns

    def test_fraud_rate(self):
        from data.generate_data import generate_fraud_data
        df = generate_fraud_data(n_samples=5000)
        assert 0.01 <= df["is_fraud"].mean() <= 0.05

    def test_no_nulls(self):
        from data.generate_data import generate_fraud_data
        df = generate_fraud_data(n_samples=500)
        assert df.isnull().sum().sum() == 0


# ── model tests ───────────────────────────────────────────────────────────────

class TestTrainingPipeline:

    def test_pipeline_builds(self):
        import sys
        sys.path.insert(0, ".")
        from model.train import build_pipeline, PARAMS
        pipeline = build_pipeline(PARAMS)
        assert "scaler" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_fit_predict(self):
        from data.generate_data import generate_fraud_data
        from model.train import build_pipeline, PARAMS
        df = generate_fraud_data(n_samples=500)
        X  = df.drop("is_fraud", axis=1)
        y  = df["is_fraud"]
        pipeline = build_pipeline(PARAMS)
        pipeline.fit(X, y)
        probas = pipeline.predict_proba(X)[:, 1]
        assert len(probas) == len(X)
        assert all(0 <= p <= 1 for p in probas)


# ── api tests ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    import sys
    sys.path.insert(0, ".")
    from api.app import app, app_state

    mock = MagicMock()
    mock.predict.return_value = np.array([0])
    mock.predict_proba.return_value = np.array([[0.95, 0.05]])
    app_state["pipeline"] = mock

    with TestClient(app) as c:
        yield c


class TestAPI:

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True

    def test_model_info(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        assert len(r.json()["features"]) == 20

    def test_predict_legit(self, client):
        r = client.post("/predict", json=LEGIT)
        assert r.status_code == 200
        data = r.json()
        assert "fraud_score" in data
        assert 0.0 <= data["fraud_score"] <= 1.0

    def test_predict_batch(self, client):
        r = client.post("/predict/batch", json={"transactions": [LEGIT, FRAUD]})
        assert r.status_code == 200
        assert r.json()["total"] == 2

    def test_metrics(self, client):
        client.post("/predict", json=LEGIT)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "fraud_rate" in r.json()

    def test_invalid_payload(self, client):
        r = client.post("/predict", json={"bad": "data"})
        assert r.status_code == 422