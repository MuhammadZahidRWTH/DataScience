from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_MANUALS = BASE_DIR / "data" / "manuals"
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss_index"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ANOMALY_THRESHOLD = 2.5
WINDOW_SIZE = 10

FORECAST_HORIZON_HOURS = 48
FAILURE_PROBABILITY_THRESHOLD = 0.75

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 5

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]

AGENT_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3
