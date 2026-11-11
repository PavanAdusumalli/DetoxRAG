"""Controlled, parameterized, read-only SQL retrieval.

Every function that returns clinical records takes a `Booking` (resolved and
validated first) and filters on BOTH PatientID and BookingNumber plus the
requested time window. There is no code path that accepts free-form SQL, and
the language model never sees or generates SQL.
"""

from __future__ import annotations

import re
import sqlite3

from backend.services.schemas import (
    ASSESSMENT_TYPE,
    Assessment,
    Booking,
    MedicationAdministration,
    MedicationOrder,
    NursingNote,
    PatientListItem,
    VitalSign,
)

PATIENT_ID_RE = re.compile(r"^P\d{3,6}$")
BOOKING_RE = re.compile(r"^B\d{4,8}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


class InvalidIdentifier(ValueError):
    pass


class RecordNotFound(LookupError):
    pass


def validate_patient_id(patient_id: str) -> str:
    if not isinstance(patient_id, str) or not PATIENT_ID_RE.match(patient_id):
        raise InvalidIdentifier(f"Invalid PatientID {patient_id!r} (expected e.g. 'P001')")
    return patient_id


def validate_booking_number(booking_number: str) -> str:
    if not isinstance(booking_number, str) or not BOOKING_RE.match(booking_number):
        raise InvalidIdentifier(f"Invalid BookingNumber {booking_number!r} (expected e.g. 'B1001')")
    return booking_number


def validate_timestamp(ts: str) -> str:
    if not isinstance(ts, str) or not TS_RE.match(ts):
        raise InvalidIdentifier(f"Invalid timestamp {ts!r} (expected YYYY-MM-DDTHH:MM:SS)")
    return ts


# --------------------------------------------------------------------------- #
# Bookings
# --------------------------------------------------------------------------- #


def list_patients(conn: sqlite3.Connection, withdrawal_type: str | None = None) -> list[PatientListItem]:
    sql = """
        SELECT b.PatientID, b.BookingNumber, b.WithdrawalType, b.AdmissionDateTime, b.ReleaseDateTime,
               COUNT(a.AssessmentID) AS n_assessments,
               MIN(a.AssessmentDateTime) AS first_assessment,
               MAX(a.AssessmentDateTime) AS last_assessment,
               (SELECT a2.Score FROM withdrawal_assessments a2
                 WHERE a2.PatientID = b.PatientID AND a2.BookingNumber = b.BookingNumber
                 ORDER BY a2.AssessmentDateTime DESC LIMIT 1) AS latest_score
        FROM patient_bookings b
        LEFT JOIN withdrawal_assessments a
               ON a.PatientID = b.PatientID AND a.BookingNumber = b.BookingNumber
        WHERE (? IS NULL OR b.WithdrawalType = ?)
        GROUP BY b.PatientID, b.BookingNumber
        ORDER BY b.PatientID, b.AdmissionDateTime"""
    rows = conn.execute(sql, (withdrawal_type, withdrawal_type)).fetchall()
    return [
        PatientListItem(**dict(r), AssessmentType=ASSESSMENT_TYPE[r["WithdrawalType"]]) for r in rows
    ]


def get_patient_bookings(conn: sqlite3.Connection, patient_id: str) -> list[Booking]:
    validate_patient_id(patient_id)
    rows = conn.execute(
        "SELECT * FROM patient_bookings WHERE PatientID = ? ORDER BY AdmissionDateTime DESC", (patient_id,)
    ).fetchall()
    return [Booking(**dict(r)) for r in rows]


def get_patient_booking(conn: sqlite3.Connection, patient_id: str, booking_number: str | None = None) -> Booking:
    """Resolve a booking for a patient.

    With `booking_number`, it must belong to `patient_id` (a booking that
    belongs to someone else is reported as not found, without revealing that
    it exists). Without it, the most recent booking is returned.
    """
    validate_patient_id(patient_id)
    if booking_number is not None:
        validate_booking_number(booking_number)
        row = conn.execute(
            "SELECT * FROM patient_bookings WHERE PatientID = ? AND BookingNumber = ?",
            (patient_id, booking_number),
        ).fetchone()
        if row is None:
            raise RecordNotFound(f"No booking {booking_number} for patient {patient_id}")
        return Booking(**dict(row))
    bookings = get_patient_bookings(conn, patient_id)
    if not bookings:
        raise RecordNotFound(f"No patient {patient_id}")
    return bookings[0]


def get_latest_event_time(conn: sqlite3.Connection, b: Booking) -> str | None:
    """Latest timestamp of any record for this booking (incl. scheduled doses)."""
    row = conn.execute(
        """SELECT MAX(t) FROM (
             SELECT MAX(AssessmentDateTime) AS t FROM withdrawal_assessments WHERE PatientID = ? AND BookingNumber = ?
             UNION ALL SELECT MAX(RecordedDateTime) FROM vital_signs WHERE PatientID = ? AND BookingNumber = ?
             UNION ALL SELECT MAX(NoteDateTime) FROM nursing_notes WHERE PatientID = ? AND BookingNumber = ?
             UNION ALL SELECT MAX(ScheduledDateTime) FROM medication_administrations WHERE PatientID = ? AND BookingNumber = ?
             UNION ALL SELECT MAX(AdministrationDateTime) FROM medication_administrations WHERE PatientID = ? AND BookingNumber = ?
           )""",
        (b.PatientID, b.BookingNumber) * 5,
    ).fetchone()
    return row[0]


# --------------------------------------------------------------------------- #
# Clinical records (always booking + window scoped)
# --------------------------------------------------------------------------- #


def _window(start: str, end: str) -> tuple[str, str]:
    validate_timestamp(start)
    validate_timestamp(end)
    if start > end:
        raise InvalidIdentifier(f"Window start {start} is after end {end}")
    return start, end


def get_withdrawal_assessments(conn: sqlite3.Connection, b: Booking, start: str, end: str) -> list[Assessment]:
    s, e = _window(start, end)
    rows = conn.execute(
        """SELECT AssessmentID, AssessmentDateTime, AssessmentType, Score FROM withdrawal_assessments
           WHERE PatientID = ? AND BookingNumber = ? AND AssessmentDateTime BETWEEN ? AND ?
           ORDER BY AssessmentDateTime, AssessmentID""",
        (b.PatientID, b.BookingNumber, s, e),
    ).fetchall()
    return [Assessment(**dict(r)) for r in rows]


def get_vital_signs(conn: sqlite3.Connection, b: Booking, start: str, end: str) -> list[VitalSign]:
    s, e = _window(start, end)
    rows = conn.execute(
        """SELECT VitalID, RecordedDateTime, SBP, DBP, Pulse, Temperature, RespiratoryRate, OxygenSaturation
           FROM vital_signs WHERE PatientID = ? AND BookingNumber = ? AND RecordedDateTime BETWEEN ? AND ?
           ORDER BY RecordedDateTime, VitalID""",
        (b.PatientID, b.BookingNumber, s, e),
    ).fetchall()
    return [VitalSign(**dict(r)) for r in rows]


def get_medication_orders(conn: sqlite3.Connection, b: Booking, start: str, end: str) -> list[MedicationOrder]:
    """Orders active at any point in the window."""
    s, e = _window(start, end)
    rows = conn.execute(
        """SELECT OrderID, MedicationName, OrderStartDateTime, OrderStopDateTime, OrderStatus
           FROM medication_orders WHERE PatientID = ? AND BookingNumber = ?
             AND OrderStartDateTime <= ? AND (OrderStopDateTime IS NULL OR OrderStopDateTime >= ?)
           ORDER BY OrderStartDateTime, OrderID""",
        (b.PatientID, b.BookingNumber, e, s),
    ).fetchall()
    return [MedicationOrder(**dict(r)) for r in rows]


def get_medication_administrations(
    conn: sqlite3.Connection, b: Booking, start: str, end: str
) -> list[MedicationAdministration]:
    """Doses *scheduled* within the window, with their documented outcome."""
    s, e = _window(start, end)
    rows = conn.execute(
        """SELECT m.AdministrationID, m.OrderID, o.MedicationName, m.ScheduledDateTime,
                  m.AdministrationDateTime, m.AdministrationStatus
           FROM medication_administrations m
           JOIN medication_orders o
             ON o.OrderID = m.OrderID AND o.PatientID = m.PatientID AND o.BookingNumber = m.BookingNumber
           WHERE m.PatientID = ? AND m.BookingNumber = ? AND m.ScheduledDateTime BETWEEN ? AND ?
           ORDER BY m.ScheduledDateTime, m.AdministrationID""",
        (b.PatientID, b.BookingNumber, s, e),
    ).fetchall()
    return [MedicationAdministration(**dict(r)) for r in rows]


def get_nursing_notes(conn: sqlite3.Connection, b: Booking, start: str, end: str) -> list[NursingNote]:
    s, e = _window(start, end)
    rows = conn.execute(
        """SELECT NoteID, NoteDateTime, NoteText FROM nursing_notes
           WHERE PatientID = ? AND BookingNumber = ? AND NoteDateTime BETWEEN ? AND ?
           ORDER BY NoteDateTime, NoteID""",
        (b.PatientID, b.BookingNumber, s, e),
    ).fetchall()
    return [NursingNote(**dict(r)) for r in rows]


def get_measurement_units(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["ColumnName"]: r["Unit"] for r in conn.execute("SELECT ColumnName, Unit FROM measurement_units")}
