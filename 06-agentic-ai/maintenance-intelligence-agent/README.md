# 🏭 MaintenanceGPT - Sovereign Agentic AI for Industrial Predictive Maintenance

> A multi-agent LLM system for industrial predictive maintenance —  
> runs fully locally via Ollama, GDPR-compliant, no cloud dependency.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
![CI](https://github.com/MuhammadZahidRWTH/DataScience-Projects/actions/workflows/maintenancegpt-ci.yml/badge.svg)

---

## 🎯 What This Project Does

MaintenanceGPT is a **5-agent LangGraph pipeline** that automates industrial maintenance intelligence:

1. Detects sensor anomalies across industrial machines
2. Diagnoses fault type using hybrid AI (Isolation Forest + PyTorch MLP)
3. Forecasts failure windows 48 hours ahead using Prophet
4. Retrieves maintenance procedures from machine manuals via RAG
5. Compiles a structured maintenance report with LLM-generated executive summary

All agents run **locally via Ollama** — no API keys, no cloud, fully GDPR-compliant.

---

## 🏗️ Architecture

```
Raw Sensor Data (15,000+ readings)
Maintenance Logs (200 entries)
         │
         ▼
┌─────────────────────┐
│   Ingestion Agent   │  • Z-score anomaly detection
│                     │  • LLM-based log cleaning (Ollama)
│                     │  • Data validation + rolling statistics
└────────┬────────────┘
         │ 298 anomalies
         ▼
┌─────────────────────┐
│   Diagnosis Agent   │  • Isolation Forest (unsupervised scoring)
│   Hybrid AI         │  • PyTorch MLP (fault classification)
│                     │  • LLM root cause explanation
└────────┬────────────┘
         │ bearing_wear / overheating
         ▼
┌─────────────────────┐
│ Forecasting Agent   │  • Prophet time-series models
│                     │  • 9 models (3 machines × 3 sensors)
│                     │  • 48h failure probability + LLM risk briefing
└────────┬────────────┘
         │ risk assessment
         ▼
┌─────────────────────┐
│  RAG Knowledge      │  • FAISS vector index over machine manuals
│  Agent              │  • HuggingFace local embeddings
│                     │  • Procedure retrieval + Ollama Q&A
└────────┬────────────┘
         │ maintenance procedures
         ▼
┌─────────────────────┐
│   Report Agent      │  • Structured JSON + Markdown report
│                     │  • Action items by priority
│                     │  • LLM executive summary for plant management
└────────┬────────────┘
         │
         ▼
  FastAPI REST API  +  Streamlit Dashboard  +  Docker
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph, LangChain |
| Local LLM | Ollama (llama3.2) |
| Unsupervised Anomaly | Isolation Forest (scikit-learn) |
| Fault Classification | PyTorch MLP |
| Time-Series Forecasting | Prophet |
| RAG / Vector Search | FAISS, HuggingFace Embeddings |
| Evaluation | RAGAS (faithfulness, answer relevancy) |
| API Serving | FastAPI, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Containerization | Docker, docker-compose |
| CI/CD | GitHub Actions |
| Language | Python 3.11+ |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- llama3.2 model pulled

```bash
ollama pull llama3.2
```

### Install

```bash
git clone https://github.com/MuhammadZahidRWTH/DataScience-Projects.git
cd DataScience-Projects/06-agentic-ai/maintenance-intelligence-agent
pip install -r requirements.txt
```

### Generate synthetic data

```bash
python data/raw/generate_data.py
```

### Run full pipeline

```bash
python pipeline.py
```

### Run individual agents

```bash
python agents/ingestion/ingestion_agent.py
python agents/diagnosis/diagnosis_agent.py
python agents/forecasting/forecasting_agent.py
python agents/rag/rag_agent.py
python agents/report/report_agent.py
```

### Start API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs: `http://localhost:8000/docs`

### Start Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard: `http://localhost:8501`

### Run with Docker

```bash
docker-compose up --build
```

---

## 📊 Pipeline Results

| Metric | Value |
|---|---|
| Sensor readings processed | 15,000 |
| Anomalies detected | 298 |
| Bearing wear faults classified | 59 |
| Overheating faults classified | 29 |
| Prophet models trained | 9 (3 machines × 3 sensors) |
| Maintenance procedures retrieved | 5 |
| Tests passing | 6/6 |

### 🔬 RAGAS Evaluation Results

RAG Knowledge Agent outputs evaluated using RAGAS metrics:

| Metric | Score | Interpretation |
|---|---|---|
| **Faithfulness** | 1.0 | Answers fully grounded in machine manual context |
| **Answer Relevancy** | 1.0 | Answers directly address the maintenance question |

```bash
python evaluation/ragas_eval.py
```

---

## 🔌 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Service health + sovereign status |
| `GET /status` | Pipeline output file status |
| `GET /report` | Latest maintenance report |
| `GET /diagnoses` | Fault diagnoses with confidence scores |
| `GET /forecasts` | 48h sensor forecasts |
| `GET /procedures` | RAG-retrieved maintenance procedures |
| `GET /anomalies` | Detected sensor anomalies |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

```
tests/test_pipeline.py::test_config_loads         PASSED
tests/test_pipeline.py::test_anomaly_detection    PASSED
tests/test_pipeline.py::test_clean_sensor_data    PASSED
tests/test_pipeline.py::test_isolation_forest     PASSED
tests/test_pipeline.py::test_api_health           PASSED
tests/test_pipeline.py::test_api_status           PASSED
```

---

## 📁 Project Structure

```
maintenance-intelligence-agent/
├── agents/
│   ├── ingestion/          # Data loading, cleaning, anomaly detection, LLM log cleaning
│   ├── diagnosis/          # Isolation Forest + PyTorch MLP + LLM explanation
│   ├── forecasting/        # Prophet time-series + LLM risk briefing
│   ├── rag/                # FAISS index + manual retrieval + Ollama Q&A
│   └── report/             # Structured report + executive summary
├── api/
│   └── main.py             # FastAPI REST service
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── evaluation/
│   └── ragas_eval.py       # RAGAS faithfulness + answer relevancy evaluation
├── data/
│   ├── raw/                # Synthetic sensor data + maintenance logs
│   ├── processed/          # Agent outputs (anomalies, diagnoses, forecasts, report, ragas scores)
│   ├── manuals/            # Machine maintenance manuals (RAG knowledge base)
│   └── faiss_index/        # Persisted FAISS vector index
├── tests/
│   └── test_pipeline.py    # 6 pytest tests
├── pipeline.py             # LangGraph full pipeline entry point
├── config.py               # Central configuration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Production container
├── docker-compose.yml      # Multi-service deployment
└── .github/workflows/      # GitHub Actions CI/CD
    └── maintenancegpt-ci.yml
```

---

## 💡 Design Decisions & Lessons Learned

**Why Isolation Forest + PyTorch instead of XGBoost?**
Isolation Forest handles unsupervised anomaly scoring where labels are scarce — realistic in manufacturing. The PyTorch MLP provides supervised classification on known fault types. Together they cover both labeled and unlabeled scenarios.

**Why Prophet for forecasting?**
Prophet handles seasonality and missing data robustly — both common in real sensor streams. The pipeline correctly shows Prophet excels at gradual trend forecasting, while discrete fault events are better handled by the Diagnosis Agent. This separation of concerns is intentional.

**Why Ollama instead of OpenAI?**
Running llama3.2 locally means no sensor data or maintenance logs leave the machine — fully GDPR-compliant and deployable in air-gapped industrial environments.

**RAGAS evaluation**
Agent outputs are evaluated using RAGAS faithfulness and answer relevancy metrics, ensuring LLM responses are grounded in retrieved context rather than hallucinated.

---

## 👤 Author

**Muhammad Zahid**
M.Sc. Data Science, RWTH Aachen University
[GitHub](https://github.com/MuhammadZahidRWTH) · [LinkedIn](https://linkedin.com/in/muhammad-zahid-772206258) · [Portfolio](https://muhammadzahidrwth.github.io)

---

## 📄 License

MIT License

---

*Sovereign Agentic AI · Made in Aachen, Germany*

