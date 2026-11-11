"""Local Qwen3-Embedding-0.6B wrapper (Sentence Transformers).

Qwen3-Embedding is asymmetric: queries are prefixed with a task instruction,
documents are embedded as-is (see the model card). Embeddings are
L2-normalized so cosine similarity == dot product.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache

import numpy as np

from backend.config import get_settings

log = logging.getLogger(__name__)

QUERY_TASK = (
    "Given a question about monitoring or managing substance withdrawal in jails "
    "and correctional health care, retrieve guideline passages that answer the question"
)


def format_query(query: str, task: str = QUERY_TASK) -> str:
    return f"Instruct: {task}\nQuery: {query}"


class Embedder:
    def __init__(self, model_path: str, device: str = "mps", batch_size: int = 8):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "mps" and not torch.backends.mps.is_available():
            log.warning("MPS unavailable; falling back to CPU for embeddings")
            device = "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        t0 = time.perf_counter()
        self.model = SentenceTransformer(
            model_path, device=device, model_kwargs={"torch_dtype": dtype}
        )
        self.device = device
        self.batch_size = batch_size
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self._lock = threading.Lock()  # one encode at a time on the GPU
        log.info("Loaded embedding model on %s in %.1fs (dim=%d)", device, time.perf_counter() - t0, self.dim)

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        with self._lock:
            vecs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )
        return vecs.astype(np.float32)

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(texts, show_progress=show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([format_query(query)])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    s = get_settings()
    if not (s.embedding_path / "config.json").exists():
        raise FileNotFoundError(
            f"Embedding model not found at {s.embedding_path}. Run `python scripts/download_models.py`."
        )
    return Embedder(str(s.embedding_path), s.embedding_device, s.embedding_batch_size)
