"""Centralized configuration for DetoxRAG.

Values are read from environment variables (optionally loaded from a `.env`
file in the project root). Relative paths are resolved against the project
root so scripts behave the same regardless of the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)

# Keep Hugging Face libraries offline by default: all models are local.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# ChromaDB: never send telemetry.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _path(name: str, default: str) -> Path:
    p = Path(os.environ.get(name, default))
    return p if p.is_absolute() else PROJECT_ROOT / p


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # Models
    llm_path: Path
    embedding_path: Path
    embedding_device: str
    embedding_batch_size: int
    llm_max_tokens: int
    llm_temperature: float

    # Data locations
    guidelines_dir: Path
    processed_dir: Path
    vector_store_dir: Path
    database_path: Path

    # Chunking
    chunk_max_words: int
    chunk_overlap_words: int

    # Retrieval
    retrieval_top_k: int
    retrieval_candidates: int
    rrf_k: int
    retrieval_min_dense_similarity: float | None

    # Timeline analytics (software settings, not clinical standards)
    review_interval_hours: float
    gap_tolerance: float
    pairing_minutes: float
    conflict_minutes: float

    @property
    def sources_manifest(self) -> Path:
        return self.guidelines_dir / "sources.json"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def chroma_dir(self) -> Path:
        return self.vector_store_dir / "chroma"

    @property
    def bm25_dir(self) -> Path:
        return self.vector_store_dir / "bm25"

    @property
    def index_info_path(self) -> Path:
        return self.vector_store_dir / "index_info.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        llm_path=_path("DETOX_LLM_PATH", "models/qwen3-4b-mlx-4bit"),
        embedding_path=_path("DETOX_EMBEDDING_PATH", "models/qwen3-embedding-0.6b"),
        embedding_device=os.environ.get("DETOX_EMBEDDING_DEVICE", "mps"),
        embedding_batch_size=_int("DETOX_EMBEDDING_BATCH_SIZE", 8),
        llm_max_tokens=_int("DETOX_LLM_MAX_TOKENS", 1024),
        llm_temperature=_float("DETOX_LLM_TEMPERATURE", 0.0),
        guidelines_dir=_path("DETOX_GUIDELINES_DIR", "data/guidelines"),
        processed_dir=_path("DETOX_PROCESSED_DIR", "data/processed"),
        vector_store_dir=_path("DETOX_VECTOR_STORE_DIR", "data/vector_store"),
        database_path=_path("DETOX_DATABASE_PATH", "data/synthetic_patients/detoxrag.db"),
        chunk_max_words=_int("DETOX_CHUNK_MAX_WORDS", 220),
        chunk_overlap_words=_int("DETOX_CHUNK_OVERLAP_WORDS", 40),
        retrieval_top_k=_int("DETOX_RETRIEVAL_TOP_K", 5),
        retrieval_candidates=_int("DETOX_RETRIEVAL_CANDIDATES", 20),
        rrf_k=_int("DETOX_RRF_K", 60),
        # Empty string disables the gate.
        retrieval_min_dense_similarity=(
            float(v) if (v := os.environ.get("DETOX_RETRIEVAL_MIN_DENSE_SIM", "0.5")) else None
        ),
        review_interval_hours=_float("DETOX_REVIEW_INTERVAL_HOURS", 4.0),
        gap_tolerance=_float("DETOX_GAP_TOLERANCE", 1.5),
        pairing_minutes=_float("DETOX_PAIRING_MINUTES", 30.0),
        conflict_minutes=_float("DETOX_CONFLICT_MINUTES", 60.0),
    )
