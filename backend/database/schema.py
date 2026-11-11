"""SQLite schema for the synthetic correctional withdrawal-monitoring dataset.

Design notes
------------
* Column names follow the project specification (PascalCase).
* Every clinical table carries (PatientID, BookingNumber) and has a composite
  foreign key to `patient_bookings`, so a record can never be attached to a
  booking that belongs to a different patient.
* Record IDs are prefixed text ("ASM-00001", "VIT-00001", ...) so they can be
  cited directly as patient-record evidence in later phases.
* Timestamps are ISO-8601 text without time zone ("2026-03-04T08:00:00"),
  representing facility local time. CHECK constraints enforce the format.
* Score ranges are enforced per instrument: CIWA-Ar 0–67, COWS 0–48. A trigger
  enforces that the instrument matches the booking's withdrawal type, so the
  two scales are never mixed for one patient.
* Vital-sign CHECKs are *data-plausibility* bounds to catch generator bugs,
  not clinical thresholds. NULL means "not recorded".
* A missing assessment is represented by the absence of a row, never by a NULL
  score: the database cannot tell us whether an assessment happened, only
  whether one was recorded.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = "1"

ISO = "'%Y-%m-%dT%H:%M:%S'"


def _iso_check(col: str, nullable: bool = False) -> str:
    # `IS` (not `=`): strftime() returns NULL for unparseable input, and a CHECK
    # that evaluates to NULL would silently pass.
    cond = f"{col} IS strftime({ISO}, {col})"
    return f"CHECK ({col} IS NULL OR {cond})" if nullable else f"CHECK ({cond})"


DDL = f"""
CREATE TABLE IF NOT EXISTS dataset_info (
    Key   TEXT PRIMARY KEY,
    Value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS measurement_units (
    TableName  TEXT NOT NULL,
    ColumnName TEXT NOT NULL,
    Unit       TEXT NOT NULL,
    Description TEXT NOT NULL,
    PRIMARY KEY (TableName, ColumnName)
);

CREATE TABLE IF NOT EXISTS patient_bookings (
    PatientID         TEXT NOT NULL,
    BookingNumber     TEXT NOT NULL PRIMARY KEY,
    AdmissionDateTime TEXT NOT NULL {_iso_check("AdmissionDateTime")},
    ReleaseDateTime   TEXT {_iso_check("ReleaseDateTime", nullable=True)},
    WithdrawalType    TEXT NOT NULL CHECK (WithdrawalType IN ('Alcohol', 'Opioid')),
    UNIQUE (PatientID, BookingNumber),
    CHECK (ReleaseDateTime IS NULL OR ReleaseDateTime > AdmissionDateTime)
);

CREATE TABLE IF NOT EXISTS withdrawal_assessments (
    AssessmentID       TEXT NOT NULL PRIMARY KEY,
    PatientID          TEXT NOT NULL,
    BookingNumber      TEXT NOT NULL,
    AssessmentDateTime TEXT NOT NULL {_iso_check("AssessmentDateTime")},
    AssessmentType     TEXT NOT NULL CHECK (AssessmentType IN ('CIWA-Ar', 'COWS')),
    Score              INTEGER NOT NULL,
    CHECK (
        (AssessmentType = 'CIWA-Ar' AND Score BETWEEN 0 AND 67) OR
        (AssessmentType = 'COWS'    AND Score BETWEEN 0 AND 48)
    ),
    FOREIGN KEY (PatientID, BookingNumber) REFERENCES patient_bookings (PatientID, BookingNumber)
);

CREATE TABLE IF NOT EXISTS vital_signs (
    VitalID          TEXT NOT NULL PRIMARY KEY,
    PatientID        TEXT NOT NULL,
    BookingNumber    TEXT NOT NULL,
    RecordedDateTime TEXT NOT NULL {_iso_check("RecordedDateTime")},
    SBP              INTEGER CHECK (SBP IS NULL OR SBP BETWEEN 60 AND 250),
    DBP              INTEGER CHECK (DBP IS NULL OR DBP BETWEEN 30 AND 150),
    Pulse            INTEGER CHECK (Pulse IS NULL OR Pulse BETWEEN 30 AND 200),
    Temperature      REAL    CHECK (Temperature IS NULL OR Temperature BETWEEN 93.0 AND 108.0),
    RespiratoryRate  INTEGER CHECK (RespiratoryRate IS NULL OR RespiratoryRate BETWEEN 6 AND 40),
    OxygenSaturation INTEGER CHECK (OxygenSaturation IS NULL OR OxygenSaturation BETWEEN 70 AND 100),
    CHECK (SBP IS NULL OR DBP IS NULL OR SBP > DBP),
    FOREIGN KEY (PatientID, BookingNumber) REFERENCES patient_bookings (PatientID, BookingNumber)
);

CREATE TABLE IF NOT EXISTS medication_orders (
    OrderID            TEXT NOT NULL PRIMARY KEY,
    PatientID          TEXT NOT NULL,
    BookingNumber      TEXT NOT NULL,
    MedicationName     TEXT NOT NULL,
    OrderStartDateTime TEXT NOT NULL {_iso_check("OrderStartDateTime")},
    OrderStopDateTime  TEXT {_iso_check("OrderStopDateTime", nullable=True)},
    OrderStatus        TEXT NOT NULL CHECK (OrderStatus IN ('Active', 'Completed', 'Discontinued')),
    CHECK (OrderStopDateTime IS NULL OR OrderStopDateTime > OrderStartDateTime),
    UNIQUE (OrderID, PatientID, BookingNumber),
    FOREIGN KEY (PatientID, BookingNumber) REFERENCES patient_bookings (PatientID, BookingNumber)
);

-- AdministrationDateTime is when the outcome was documented. It is NULL exactly
-- when the status is 'Not documented' (a scheduled dose with no record), which
-- is distinct from a documented 'Not administered'.
CREATE TABLE IF NOT EXISTS medication_administrations (
    AdministrationID       TEXT NOT NULL PRIMARY KEY,
    OrderID                TEXT NOT NULL,
    PatientID              TEXT NOT NULL,
    BookingNumber          TEXT NOT NULL,
    ScheduledDateTime      TEXT NOT NULL {_iso_check("ScheduledDateTime")},
    AdministrationDateTime TEXT {_iso_check("AdministrationDateTime", nullable=True)},
    AdministrationStatus   TEXT NOT NULL CHECK (AdministrationStatus IN
        ('Administered', 'Not administered', 'Refused', 'Held', 'Not documented')),
    CHECK ((AdministrationStatus = 'Not documented') = (AdministrationDateTime IS NULL)),
    FOREIGN KEY (OrderID, PatientID, BookingNumber)
        REFERENCES medication_orders (OrderID, PatientID, BookingNumber),
    FOREIGN KEY (PatientID, BookingNumber) REFERENCES patient_bookings (PatientID, BookingNumber)
);

CREATE TABLE IF NOT EXISTS nursing_notes (
    NoteID        TEXT NOT NULL PRIMARY KEY,
    PatientID     TEXT NOT NULL,
    BookingNumber TEXT NOT NULL,
    NoteDateTime  TEXT NOT NULL {_iso_check("NoteDateTime")},
    NoteText      TEXT NOT NULL CHECK (length(trim(NoteText)) > 0),
    FOREIGN KEY (PatientID, BookingNumber) REFERENCES patient_bookings (PatientID, BookingNumber)
);

-- Generator ground truth (scenario labels, injected defects) used ONLY by tests
-- and evaluation. It is not part of the simulated health record; application
-- retrieval code must never read it or pass it to the language model.
CREATE TABLE IF NOT EXISTS synthetic_ground_truth (
    PatientID     TEXT NOT NULL,
    BookingNumber TEXT NOT NULL PRIMARY KEY,
    Scenario      TEXT NOT NULL,
    Trajectory    TEXT NOT NULL,
    Details       TEXT NOT NULL,  -- JSON
    FOREIGN KEY (PatientID, BookingNumber) REFERENCES patient_bookings (PatientID, BookingNumber)
);

CREATE TRIGGER IF NOT EXISTS trg_assessment_type_matches_booking
BEFORE INSERT ON withdrawal_assessments
FOR EACH ROW
WHEN NEW.AssessmentType <> (
    SELECT CASE WithdrawalType WHEN 'Alcohol' THEN 'CIWA-Ar' WHEN 'Opioid' THEN 'COWS' END
    FROM patient_bookings WHERE BookingNumber = NEW.BookingNumber
)
BEGIN
    SELECT RAISE(ABORT, 'AssessmentType does not match booking WithdrawalType');
END;

CREATE INDEX IF NOT EXISTS idx_assess_booking_time ON withdrawal_assessments (BookingNumber, AssessmentDateTime);
CREATE INDEX IF NOT EXISTS idx_vitals_booking_time ON vital_signs (BookingNumber, RecordedDateTime);
CREATE INDEX IF NOT EXISTS idx_orders_booking ON medication_orders (BookingNumber);
CREATE INDEX IF NOT EXISTS idx_mar_booking_time ON medication_administrations (BookingNumber, ScheduledDateTime);
CREATE INDEX IF NOT EXISTS idx_notes_booking_time ON nursing_notes (BookingNumber, NoteDateTime);
CREATE INDEX IF NOT EXISTS idx_bookings_patient ON patient_bookings (PatientID);
"""

UNITS = [
    ("vital_signs", "SBP", "mmHg", "Systolic blood pressure"),
    ("vital_signs", "DBP", "mmHg", "Diastolic blood pressure"),
    ("vital_signs", "Pulse", "beats/min", "Heart rate"),
    ("vital_signs", "Temperature", "°F", "Body temperature"),
    ("vital_signs", "RespiratoryRate", "breaths/min", "Respiratory rate"),
    ("vital_signs", "OxygenSaturation", "%", "Peripheral oxygen saturation (SpO2)"),
    ("withdrawal_assessments", "Score", "points", "Total score: CIWA-Ar 0–67, COWS 0–48 (not interchangeable)"),
]

CLINICAL_TABLES = [
    "patient_bookings",
    "withdrawal_assessments",
    "vital_signs",
    "medication_orders",
    "medication_administrations",
    "nursing_notes",
]


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL)
    conn.executemany("INSERT OR REPLACE INTO measurement_units VALUES (?, ?, ?, ?)", UNITS)
