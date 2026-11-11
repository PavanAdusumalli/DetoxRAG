"""Request/response schemas specific to the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from backend.services.schemas import Booking

RetrievalMode = Literal["hybrid", "dense", "bm25", "none"]


class AskRequest(BaseModel):
    patient_id: str = Field(pattern=r"^P\d{3,6}$", examples=["P015"])
    booking_number: str | None = Field(default=None, pattern=r"^B\d{4,8}$", examples=["B1015"])
    question: str = Field(min_length=1, max_length=1000,
                          examples=["Summarize this patient's withdrawal progression over the last 12 hours."])
    hours: float | None = Field(default=None, gt=0, le=720, description="Review the last N hours; omit for the whole booking")
    retrieval_mode: RetrievalMode = "hybrid"
    k: int = Field(default=4, ge=1, le=8)


class GuidelineAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["How often should CIWA-Ar be reassessed?"])
    method: Literal["hybrid", "dense", "bm25"] = "hybrid"
    k: int = Field(default=5, ge=1, le=10)


class PatientDetail(BaseModel):
    synthetic_data_notice: str = "SYNTHETIC DATA — fictional patient for software demonstration only."
    booking: Booking
    other_bookings: list[str]
    monitoring_type: str
    record_counts: dict[str, int]
    first_record: str | None
    latest_record: str | None


class HealthResponse(BaseModel):
    status: str
    version: str
    database: dict
    guideline_index: dict
    models: dict
