"""config/settings.py — central config loaded from .env"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_backend: str = Field("ollama", description="openai | ollama")
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: str = "data/faiss_index"

    # Retrieval
    top_k_vector: int = 20
    top_k_graph_hops: int = 2
    final_k: int = 6

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


settings = Settings()
