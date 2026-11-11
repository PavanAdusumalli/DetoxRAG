from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend import __version__
from backend.api.deps import models_loaded
from backend.api.schemas.requests import HealthResponse
from backend.config import PROJECT_ROOT, get_settings

router = APIRouter(tags=["system"])

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


@router.get("/api/health", response_model=HealthResponse)
def health():
    """Liveness plus availability of data, indexes and models (does not load models)."""
    s = get_settings()
    db_ok = s.database_path.exists()
    db = {"present": db_ok, "path": str(s.database_path.relative_to(PROJECT_ROOT)) if db_ok else None}
    if db_ok:
        from backend.database.db import connect_readonly

        c = connect_readonly()
        db["bookings"] = c.execute("SELECT COUNT(*) FROM patient_bookings").fetchone()[0]
        db["notice"] = c.execute("SELECT Value FROM dataset_info WHERE Key = 'notice'").fetchone()[0]
        c.close()
    idx = {"present": s.chunks_path.exists() and s.chroma_dir.exists() and s.bm25_dir.exists()}
    if s.index_info_path.exists():
        info = json.loads(s.index_info_path.read_text())
        idx.update(n_chunks=info["n_chunks"], built_at=info["built_at"])
    models = {
        "llm_present": (s.llm_path / "config.json").exists(),
        "embedding_present": (s.embedding_path / "config.json").exists(),
        **{f"{k}_loaded": v for k, v in models_loaded().items()},
    }
    ok = db_ok and idx["present"] and models["llm_present"] and models["embedding_present"]
    return HealthResponse(status="ok" if ok else "degraded", version=__version__, database=db,
                          guideline_index=idx, models=models)


@router.get("/api/evaluation/results", tags=["evaluation"])
def evaluation_results():
    """Latest benchmark results produced by `python -m evaluation.run_evaluation`."""
    path = RESULTS_DIR / "latest.json"
    if not path.exists():
        raise HTTPException(404, "No evaluation results yet. Run `python -m evaluation.run_evaluation`.")
    return json.loads(path.read_text())
