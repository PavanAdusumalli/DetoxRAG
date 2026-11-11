"""Integrity checks for the synthetic database (shared by tests and scripts)."""

from __future__ import annotations

import sqlite3

from backend.database.schema import CLINICAL_TABLES

# Each check is a query returning offending rows; an empty result means pass.
CHECKS: dict[str, str] = {
    "assessment type matches withdrawal type": """
        SELECT a.AssessmentID FROM withdrawal_assessments a
        JOIN patient_bookings b USING (BookingNumber)
        WHERE (b.WithdrawalType = 'Alcohol' AND a.AssessmentType <> 'CIWA-Ar')
           OR (b.WithdrawalType = 'Opioid'  AND a.AssessmentType <> 'COWS')""",
    "patient/booking pairs exist (assessments)": """
        SELECT AssessmentID FROM withdrawal_assessments a
        WHERE NOT EXISTS (SELECT 1 FROM patient_bookings b
                          WHERE b.PatientID = a.PatientID AND b.BookingNumber = a.BookingNumber)""",
    "records not before admission": """
        SELECT 'ASM ' || AssessmentID FROM withdrawal_assessments x JOIN patient_bookings b USING (BookingNumber)
            WHERE x.AssessmentDateTime < b.AdmissionDateTime
        UNION ALL SELECT 'VIT ' || VitalID FROM vital_signs x JOIN patient_bookings b USING (BookingNumber)
            WHERE x.RecordedDateTime < b.AdmissionDateTime
        UNION ALL SELECT 'NOTE ' || NoteID FROM nursing_notes x JOIN patient_bookings b USING (BookingNumber)
            WHERE x.NoteDateTime < b.AdmissionDateTime
        UNION ALL SELECT 'ORD ' || OrderID FROM medication_orders x JOIN patient_bookings b USING (BookingNumber)
            WHERE x.OrderStartDateTime < b.AdmissionDateTime""",
    "records not after release": """
        SELECT 'ASM ' || AssessmentID FROM withdrawal_assessments x JOIN patient_bookings b USING (BookingNumber)
            WHERE b.ReleaseDateTime IS NOT NULL AND x.AssessmentDateTime > b.ReleaseDateTime
        UNION ALL SELECT 'VIT ' || VitalID FROM vital_signs x JOIN patient_bookings b USING (BookingNumber)
            WHERE b.ReleaseDateTime IS NOT NULL AND x.RecordedDateTime > b.ReleaseDateTime
        UNION ALL SELECT 'NOTE ' || NoteID FROM nursing_notes x JOIN patient_bookings b USING (BookingNumber)
            WHERE b.ReleaseDateTime IS NOT NULL AND x.NoteDateTime > b.ReleaseDateTime
        UNION ALL SELECT 'MAR ' || AdministrationID FROM medication_administrations x JOIN patient_bookings b USING (BookingNumber)
            WHERE b.ReleaseDateTime IS NOT NULL AND (x.ScheduledDateTime > b.ReleaseDateTime
                  OR x.AdministrationDateTime > b.ReleaseDateTime)""",
    "scheduled doses within order window": """
        SELECT m.AdministrationID FROM medication_administrations m JOIN medication_orders o USING (OrderID)
        WHERE m.ScheduledDateTime < o.OrderStartDateTime
           OR (o.OrderStopDateTime IS NOT NULL AND m.ScheduledDateTime >= o.OrderStopDateTime)""",
    "administration documented no earlier than 1 h before schedule": """
        SELECT AdministrationID FROM medication_administrations
        WHERE AdministrationDateTime IS NOT NULL
          AND AdministrationDateTime < datetime(ScheduledDateTime, '-1 hour')""",
    "administration belongs to same booking as its order": """
        SELECT m.AdministrationID FROM medication_administrations m JOIN medication_orders o USING (OrderID)
        WHERE m.BookingNumber <> o.BookingNumber OR m.PatientID <> o.PatientID""",
    "every booking has at least one assessment": """
        SELECT BookingNumber FROM patient_bookings b
        WHERE NOT EXISTS (SELECT 1 FROM withdrawal_assessments a WHERE a.BookingNumber = b.BookingNumber)""",
    "assessment IDs chronological within booking": """
        SELECT a1.AssessmentID FROM withdrawal_assessments a1 JOIN withdrawal_assessments a2
          ON a1.BookingNumber = a2.BookingNumber AND a1.AssessmentID < a2.AssessmentID
        WHERE a1.AssessmentDateTime > a2.AssessmentDateTime""",
}


def run_integrity_checks(conn: sqlite3.Connection) -> dict[str, list]:
    """Return {check_name: offending_rows}; all lists empty means the DB is valid."""
    results: dict[str, list] = {}
    results["sqlite integrity_check"] = [
        r[0] for r in conn.execute("PRAGMA integrity_check") if r[0] != "ok"
    ]
    results["foreign_key_check"] = [tuple(r) for r in conn.execute("PRAGMA foreign_key_check")]
    for name, sql in CHECKS.items():
        results[name] = [r[0] for r in conn.execute(sql)]
    return results


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in CLINICAL_TABLES}
