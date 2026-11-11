"""FastAPI dependencies. Overridable in tests via `app.dependency_overrides`."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator

from backend.database.db import connect_readonly

_retrievers: dict[str, object] = {}
_lock = threading.Lock()


def get_db() -> Iterator[sqlite3.Connection]:
    """A fresh read-only connection per request."""
    conn = connect_readonly()
    try:
        yield conn
    finally:
        conn.close()


def get_llm():
    from backend.rag.llm import get_llm as _get

    return _get()


def get_retriever_factory():
    """Return a callable mode -> retriever (cached per mode)."""

    def factory(mode: str):
        if mode == "none":
            return None
        with _lock:
            if mode not in _retrievers:
                from backend.rag.retrieval import get_retriever

                _retrievers[mode] = get_retriever(mode)
            return _retrievers[mode]

    return factory


def models_loaded() -> dict[str, bool]:
    from backend.rag.embeddings import get_embedder
    from backend.rag.llm import get_llm as _llm

    return {"llm": _llm.cache_info().currsize > 0, "embedding": get_embedder.cache_info().currsize > 0}
