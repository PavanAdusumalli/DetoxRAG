from __future__ import annotations

import logging
import sqlite3

from fastapi import APIRouter, Depends

from backend.api.deps import get_db, get_llm, get_retriever_factory
from backend.api.schemas.requests import AskRequest, GuidelineAskRequest
from backend.rag.generation import GuidelineAnswer, answer_question
from backend.services.clinical_summary import ClinicalSummaryResponse, generate_clinical_summary

log = logging.getLogger("detoxrag.api.rag")
router = APIRouter(tags=["assistant"])


@router.post("/api/rag/ask", response_model=ClinicalSummaryResponse)
def ask_about_patient(
    req: AskRequest,
    conn: sqlite3.Connection = Depends(get_db),
    llm=Depends(get_llm),
    retrievers=Depends(get_retriever_factory),
):
    """Grounded review summary: SQL records + timeline analytics + guideline RAG + local Qwen3."""
    res = generate_clinical_summary(
        conn, req.patient_id, req.question, llm,
        retriever=retrievers(req.retrieval_mode), booking_number=req.booking_number,
        hours=req.hours, retrieval_mode=req.retrieval_mode, k=req.k,
    )
    log.info(
        "rag_ask patient=%s booking=%s mode=%s passages=%d patient_cites=%d guideline_cites=%d "
        "invalid=%d fact_fail=%d total_ms=%.0f",
        res.patient_id, res.booking_number, res.retrieval_mode, len(res.retrieved_passages),
        len(res.patient_evidence), len(res.guideline_citations), len(res.validation.invalid_citations),
        len(res.validation.fact_check_failures), res.timings_ms.get("total", 0),
    )
    return res


@router.post("/api/guidelines/ask", response_model=GuidelineAnswer)
def ask_guidelines(req: GuidelineAskRequest, llm=Depends(get_llm), retrievers=Depends(get_retriever_factory)):
    """Guideline-only question answering with verified citations."""
    return answer_question(req.question, retrievers(req.method), llm, k=req.k)
