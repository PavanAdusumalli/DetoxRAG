from __future__ import annotations

import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, Path, Query

from backend.api.deps import get_db
from backend.api.schemas.requests import PatientDetail
from backend.database import queries as q
from backend.services.schemas import PatientListItem, PatientTimeline
from backend.services.timeline import build_patient_timeline

router = APIRouter(prefix="/api/patients", tags=["patients"])

PatientIdPath = Path(pattern=r"^P\d{3,6}$", examples=["P015"])
BookingQuery = Query(default=None, pattern=r"^B\d{4,8}$")


@router.get("", response_model=list[PatientListItem])
def list_patients(
    withdrawal_type: Literal["Alcohol", "Opioid"] | None = None,
    conn: sqlite3.Connection = Depends(get_db),
):
    """All synthetic patients with booking summary (one row per booking)."""
    return q.list_patients(conn, withdrawal_type)


@router.get("/{patient_id}", response_model=PatientDetail)
def get_patient(
    patient_id: str = PatientIdPath,
    booking_number: str | None = BookingQuery,
    conn: sqlite3.Connection = Depends(get_db),
):
    b = q.get_patient_booking(conn, patient_id, booking_number)
    others = [x.BookingNumber for x in q.get_patient_bookings(conn, patient_id) if x.BookingNumber != b.BookingNumber]
    counts = {}
    for table in ("withdrawal_assessments", "vital_signs", "medication_orders", "medication_administrations", "nursing_notes"):
        counts[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE PatientID = ? AND BookingNumber = ?",  # table names are constants
            (b.PatientID, b.BookingNumber),
        ).fetchone()[0]
    first = conn.execute(
        "SELECT MIN(AssessmentDateTime) FROM withdrawal_assessments WHERE PatientID = ? AND BookingNumber = ?",
        (b.PatientID, b.BookingNumber),
    ).fetchone()[0]
    return PatientDetail(
        booking=b, other_bookings=others, monitoring_type=b.assessment_type, record_counts=counts,
        first_record=first, latest_record=q.get_latest_event_time(conn, b),
    )


@router.get("/{patient_id}/timeline", response_model=PatientTimeline)
def get_timeline(
    patient_id: str = PatientIdPath,
    booking_number: str | None = BookingQuery,
    hours: float | None = Query(default=None, gt=0, le=720, description="Last N hours of recorded monitoring"),
    start: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"),
    end: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Records + deterministic analytics for one booking and time window."""
    return build_patient_timeline(conn, patient_id, booking_number, hours=hours, start=start, end=end)
