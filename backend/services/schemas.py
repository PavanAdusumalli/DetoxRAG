"""Data models for patient records and deterministic timeline analytics.

Record models mirror the SQLite column names (PascalCase) so every value can
be traced to its source row. Analytics models use snake_case.

Wording convention: anything absent is "not recorded in the retrieved
dataset" — never "not done". The database only shows what was documented.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SCALE_MAX = {"CIWA-Ar": 67, "COWS": 48}
ASSESSMENT_TYPE = {"Alcohol": "CIWA-Ar", "Opioid": "COWS"}


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


class Booking(BaseModel):
    PatientID: str
    BookingNumber: str
    AdmissionDateTime: str
    ReleaseDateTime: str | None
    WithdrawalType: str

    @property
    def assessment_type(self) -> str:
        return ASSESSMENT_TYPE[self.WithdrawalType]


class Assessment(BaseModel):
    AssessmentID: str
    AssessmentDateTime: str
    AssessmentType: str
    Score: int


class VitalSign(BaseModel):
    VitalID: str
    RecordedDateTime: str
    SBP: int | None
    DBP: int | None
    Pulse: int | None
    Temperature: float | None
    RespiratoryRate: int | None
    OxygenSaturation: int | None


class MedicationOrder(BaseModel):
    OrderID: str
    MedicationName: str
    OrderStartDateTime: str
    OrderStopDateTime: str | None
    OrderStatus: str


class MedicationAdministration(BaseModel):
    AdministrationID: str
    OrderID: str
    MedicationName: str
    ScheduledDateTime: str
    AdministrationDateTime: str | None
    AdministrationStatus: str


class NursingNote(BaseModel):
    NoteID: str
    NoteDateTime: str
    NoteText: str


class PatientListItem(BaseModel):
    PatientID: str
    BookingNumber: str
    WithdrawalType: str
    AssessmentType: str
    AdmissionDateTime: str
    ReleaseDateTime: str | None
    n_assessments: int
    first_assessment: str | None
    last_assessment: str | None
    latest_score: int | None


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #


class TimeWindow(BaseModel):
    start: str
    end: str
    hours: float
    label: str
    end_reference: str = Field(
        description="How the window end was chosen, e.g. 'latest recorded event for this booking'"
    )


class ScorePoint(BaseModel):
    assessment_id: str
    time: str
    score: int


class ScoreSummary(BaseModel):
    assessment_type: str
    scale_max: int
    n_assessments: int
    earliest: ScorePoint | None
    latest: ScorePoint | None
    change: int | None = Field(description="latest - earliest recorded score")
    minimum: ScorePoint | None
    maximum: ScorePoint | None
    net_direction: str = Field(
        description="Descriptive only: increasing | decreasing | unchanged | single assessment | no assessments recorded"
    )
    n_increases: int
    n_decreases: int
    direction_changes: int = Field(description="Times the step-to-step direction reversed")
    observation_span_hours: float | None


class Interval(BaseModel):
    from_id: str | None
    to_id: str | None
    start: str
    end: str
    hours: float


class IntervalSummary(BaseModel):
    n_intervals: int
    intervals: list[Interval]
    min_hours: float | None
    max_hours: float | None
    mean_hours: float | None
    median_hours: float | None
    configured_review_interval_hours: float = Field(
        description="Software setting used to flag long intervals; not a clinical standard"
    )
    flag_threshold_hours: float
    long_intervals: list[Interval]
    leading_gap: Interval | None = Field(description="Window start to first assessment, if longer than the threshold")
    trailing_gap: Interval | None = Field(description="Last assessment to window end, if longer than the threshold")


class ValuePoint(BaseModel):
    record_id: str
    time: str
    value: float


class VitalMeasureSummary(BaseModel):
    measure: str
    unit: str
    n_recorded: int
    n_not_recorded: int
    earliest: ValuePoint | None
    latest: ValuePoint | None
    minimum: ValuePoint | None
    maximum: ValuePoint | None
    change: float | None


class VitalsSummary(BaseModel):
    n_sets: int
    measures: list[VitalMeasureSummary]


class OrderSummary(BaseModel):
    order: MedicationOrder
    n_scheduled_in_window: int
    status_counts: dict[str, int]


class MedicationSummary(BaseModel):
    n_scheduled_in_window: int
    status_counts: dict[str, int]
    orders: list[OrderSummary]
    not_administered: list[MedicationAdministration] = Field(
        description="Scheduled doses whose documented status is not 'Administered' (incl. 'Not documented')"
    )


class DocumentationGap(BaseModel):
    gap_type: str
    description: str
    start: str | None = None
    end: str | None = None
    record_ids: list[str] = Field(default_factory=list)


class DocumentationConflict(BaseModel):
    conflict_type: str
    description: str
    time: str
    record_ids: list[str]


class TimelineEvent(BaseModel):
    time: str
    kind: str  # assessment | vitals | medication | note
    record_id: str
    summary: str


class PatientTimeline(BaseModel):
    synthetic_data_notice: str = "SYNTHETIC DATA — fictional patient for software demonstration only."
    booking: Booking
    window: TimeWindow
    units: dict[str, str]
    assessments: list[Assessment]
    vitals: list[VitalSign]
    orders: list[MedicationOrder]
    administrations: list[MedicationAdministration]
    notes: list[NursingNote]
    score_summary: ScoreSummary
    intervals: IntervalSummary
    vitals_summary: VitalsSummary
    medication_summary: MedicationSummary
    documentation_gaps: list[DocumentationGap]
    documentation_conflicts: list[DocumentationConflict]
    events: list[TimelineEvent]
