from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from backend.api.deps import get_retriever_factory
from backend.config import get_settings
from backend.rag.models import Chunk, RetrievedPassage

router = APIRouter(prefix="/api/guidelines", tags=["guidelines"])


@router.get("")
def list_guidelines():
    """Indexed sources, chunk counts, excluded sources and index build info."""
    s = get_settings()
    manifest = json.loads(s.sources_manifest.read_text())
    info = json.loads(s.index_info_path.read_text()) if s.index_info_path.exists() else None
    per_doc = (info or {}).get("chunks_per_document", {})
    docs = []
    for d in manifest["documents"]:
        docs.append({
            "document_id": d["document_id"], "title": d["title"], "short_title": d["short_title"],
            "publisher": d["publisher"], "publication_date": d["publication_date"],
            "document_version": d.get("document_version"), "source_url": d["source_url"],
            "usage_note": d.get("usage_note"), "chunks_indexed": per_doc.get(d["document_id"], 0),
            "file_present": (s.guidelines_dir / d["file"]).exists(),
        })
    return {"documents": docs, "not_included": manifest.get("not_included", []), "index": info}


@router.get("/passages/{chunk_id}", response_model=Chunk)
def get_passage(chunk_id: str = Path(pattern=r"^[A-Z]{2,5}-\d{4}-\d{4}$")):
    """Full text and citation metadata of one indexed passage."""
    from backend.rag.retrieval import get_chunk_store

    store = get_chunk_store()
    if chunk_id not in store:
        raise HTTPException(404, f"No passage {chunk_id}")
    return store.get(chunk_id)


@router.get("/search", response_model=list[RetrievedPassage])
def search(
    q: str = Query(min_length=1, max_length=500),
    method: Literal["hybrid", "dense", "bm25"] = "hybrid",
    k: int = Query(default=5, ge=1, le=20),
    retrievers=Depends(get_retriever_factory),
):
    """Retrieval only (no generation) — for inspecting what each method returns."""
    return retrievers(method).search(q, k)
