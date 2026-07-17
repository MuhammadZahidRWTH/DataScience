# 💳 Fraud Detection — End-to-End MLOps Pipeline

> Production-grade fraud detection system with full MLOps lifecycle:  
> training, experiment tracking, containerized serving, and live drift monitoring.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-ROC--AUC_0.88-orange?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square)
![CI](https://github.com/MuhammadZahidRWTH/DataScience-Projects/actions/workflows/ci-cd.yml/badge.svg)

---

## 🎯 What This Project Does

A complete MLOps pipeline for financial transaction fraud detection:

1. **Generates** realistic synthetic transaction data (10,000 transactions, 2.4% fraud rate)
2. **Trains** XGBoost classifier with SMOTE oversampling and threshold tuning
3. **Tracks** all experiments, metrics, and artifacts with MLflow
4. **Serves** predictions via FastAPI at sub-10ms latency
5. **Monitors** production data drift weekly using Evidently AI
6. **Automates** model quality gates via GitHub Actions CI/CD

---

## 🏗️ Architecture

```
Synthetic Transaction Data (10,000 rows)
           │
           ▼
┌──────────────────────┐
│   Data Generation    │  • 2.4% fraud rate (realistic imbalance)
│                      │  • Feature engineering: amount, time, merchant
│                      │  • Drift dataset for monitoring tests
└─────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│   Model Training     │  • XGBoost with scale_pos_weight=49
│                      │  • SMOTE oversampling on minority class
│                      │  • 5-fold stratified cross-validation
│                      │  • Threshold tuning for precision/recall balance
└─────────┬────────────┘
           │ ROC-AUC 0.88
           ▼
┌──────────────────────┐
│   MLflow Tracking    │  • Experiment logging: params, metrics, artifacts
│                      │  • Model registry with versioning
│                      │  • Classification report + feature importance
└─────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│   FastAPI Serving    │  • /predict endpoint at sub-10ms latency
│                      │  • /health and /metrics endpoints
│                      │  • Batch prediction support
└─────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│   Evidently AI       │  • Data drift detection on feature distributions
│                      │  • Weekly monitoring reports
│                      │  • Automatic alerts on distribution shift
└──────────────────────┘
           │
           ▼
  Docker + GitHub Actions CI/CD
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost, scikit-learn Pipeline |
| Imbalanced Data | SMOTE (imbalanced-learn) |
| Experiment Tracking | MLflow |
| API Serving | FastAPI, Uvicorn |
| Drift Monitoring | Evidently AI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Language | Python 3.11+ |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | **0.88** |
| Average Precision | 0.63 |
| F1 Score | 0.66 |
| Precision | 0.84 |
| Recall | 0.54 |
| CV ROC-AUC | 0.86 ± 0.03 |

**Dataset:** 10,000 transactions · 2.4% fraud rate · 80/20 train/test split

---

## 🚀 Quickstart

### Install

```bash
git clone https://github.com/MuhammadZahidRWTH/DataScience-Projects.git
cd DataScience-Projects/03-mlops/fraud-detection
pip install -r requirements.txt
```

### Generate data

```bash
python data/generate.py
```

### Train model

```bash
python model/train.py
```

### View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000`

### Start API

```bash
uvicorn api.main:app --reload --port 8000
```

### Run with Docker

```bash
docker-compose up --build
```

### Run drift monitoring

```bash
python monitoring/drift_report.py
```

---

## 🔌 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Service health check |
| `POST /predict` | Single transaction fraud prediction |
| `POST /predict/batch` | Batch prediction |
| `GET /metrics` | Model performance metrics |

### Sample prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 2500.0, "hour": 2, "day_of_week": 6}'
```

```json
{
  "fraud_probability": 0.847,
  "prediction": 1,
  "label": "FRAUD",
  "latency_ms": 8.2
}
```

---

## 💡 Design Decisions

**Why `scale_pos_weight=49`?**
Class ratio is 98:2 (legit:fraud), so 98/2 = 49. This tells XGBoost to weight the minority fraud class 49x higher during training — more effective than simple oversampling alone.

**Why SMOTE + threshold tuning?**
SMOTE generates synthetic fraud samples to balance the training set. Threshold tuning (default 0.5 → optimized) then adjusts the decision boundary to balance precision and recall for the specific business cost of false negatives vs false positives.

**Why MLflow over manual logging?**
Every experiment run — including failed ones — is automatically tracked with full reproducibility. This makes it trivial to compare 50 hyperparameter combinations and roll back to any previous model version.

**Why Evidently AI for monitoring?**
Evidently generates HTML drift reports that non-technical stakeholders can read. It monitors feature distributions in production vs training, catching data drift before it silently degrades model performance.

---

## 📁 Project Structure

```
fraud-detection/
├── data/
│   ├── generate.py          # Synthetic transaction data generator
│   ├── transactions.csv     # Training data (10,000 rows)
│   └── drift_data.csv       # Drifted data for monitoring tests
├── model/
│   ├── train.py             # XGBoost training + MLflow logging
│   ├── pipeline.pkl         # Saved sklearn pipeline
│   ├── classification_report.txt
│   └── feature_importance.json
├── api/
│   └── main.py              # FastAPI prediction service
├── monitoring/
│   └── drift_report.py      # Evidently AI drift detection
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/
    └── ci-cd.yml            # GitHub Actions CI/CD
```

---

## 👤 Author

**Muhammad Zahid**
M.Sc. Data Science, RWTH Aachen University
[GitHub](https://github.com/MuhammadZahidRWTH) · [LinkedIn](https://linkedin.com/in/muhammad-zahid-772206258) · [Portfolio](https://muhammadzahidrwth.github.io)

---

## 📄 License

MIT License

---

*End-to-End MLOps · Made in Aachen, Germany*
