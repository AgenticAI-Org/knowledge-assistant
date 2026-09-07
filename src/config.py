"""Central application configuration, loaded from environment variables / .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # env_file is anchored to PROJECT_ROOT (not the CWD) so `python -m scripts.build_index`
    # picks up the key regardless of which directory it is invoked from.
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_persist_dir: Path = PROJECT_ROOT / "storage" / "chroma"
    bm25_index_path: Path = PROJECT_ROOT / "storage" / "bm25_corpus.json"
    log_dir: Path = PROJECT_ROOT / "logs"

    # --- Chunking ---
    # Sized so the corpus yields comfortably more chunks than top_k_vector/top_k_bm25.
    # If k >= the total chunk count, every query retrieves the whole corpus and the
    # retrieval eval degenerates to Hit Rate 1.0 for every method -- see docs/EVALUATION.md.
    chunk_size: int = 400
    chunk_overlap: int = 80

    # --- Retrieval ---
    top_k_vector: int = 10
    top_k_bm25: int = 10
    rrf_k: int = 60
    top_k_rerank: int = 4
    # Broad 'summarise/list/overview' queries keep a wider window: answering them
    # from a factoid-sized context silently omits most of the policy.
    top_k_rerank_broad: int = 12
    # k at which all retrieval arms are compared in the eval, so vector-only,
    # BM25-only and hybrid+rerank are scored over the same number of candidates.
    eval_top_k: int = 4
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Applied to factoid queries only. Cross-encoder scores are not comparable across
    # query types, so this threshold is meaningless for broad queries -- see
    # src/retrieval/query_intent.py.
    rerank_score_floor: float = -3.0  # below this, treat context as too weak to answer from

    # --- Memory ---
    memory_window_turns: int = 6

    # --- Logging ---
    log_level: str = "INFO"
    log_json: bool = True

    # --- Ingestion ---
    embed_batch_size: int = 64

    # --- Collection naming ---
    chroma_collection_name: str = "employee_knowledge_base"


settings = Settings()
