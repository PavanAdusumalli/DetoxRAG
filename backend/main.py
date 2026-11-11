"""DetoxRAG FastAPI application.

Run:
    uvicorn backend.main:app --port 8000            # API (+ built frontend if present)
    DETOX_PRELOAD_MODELS=1 uvicorn backend.main:app  # load models at startup

Interactive API docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from backend import __version__
from backend.api.routes import guidelines, patients, rag, system
from backend.config import PROJECT_ROOT
from backend.database.queries import InvalidIdentifier, RecordNotFound

log = logging.getLogger("detoxrag.api")


class FrontendFiles(StaticFiles):
    """Serve the built UI. index.html must not be cached: it names the
    content-hashed JS/CSS bundles, so a stale copy loads an old build."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        if path in ("", ".", "index.html") or not path.startswith("assets/"):
            response.headers["Cache-Control"] = "no-cache"
        else:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("DETOX_LOG_LEVEL", "INFO"),
            format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    if os.environ.get("DETOX_PRELOAD_MODELS") == "1":
        from backend.api.deps import get_llm, get_retriever_factory

        t0 = time.perf_counter()
        get_retriever_factory()("hybrid")
        get_llm()
        log.info("event=models_preloaded seconds=%.1f", time.perf_counter() - t0)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="DetoxRAG API",
        version=__version__,
        description=(
            "AI-assisted withdrawal-monitoring review over **synthetic** patient data and public guidelines. "
            "Research prototype — not a clinical decision-support system."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("request_id=%s method=%s path=%s status=500", rid, request.method, request.url.path)
            raise
        ms = (time.perf_counter() - t0) * 1000
        response.headers["x-request-id"] = rid
        if request.url.path.startswith("/api"):
            log.info("request_id=%s method=%s path=%s status=%d duration_ms=%.0f",
                     rid, request.method, request.url.path, response.status_code, ms)
        return response

    @app.exception_handler(InvalidIdentifier)
    async def _invalid(_: Request, exc: InvalidIdentifier):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(RecordNotFound)
    async def _not_found(_: Request, exc: RecordNotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def _unavailable(_: Request, exc: FileNotFoundError):
        log.error("event=resource_unavailable detail=%s", exc)
        return JSONResponse(status_code=503, content={"detail": f"Required resource unavailable: {exc}"})

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    for r in (system.router, patients.router, rag.router, guidelines.router):
        app.include_router(r)

    dist = PROJECT_ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/", FrontendFiles(directory=dist, html=True), name="frontend")
    return app


app = create_app()
