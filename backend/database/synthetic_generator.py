"""Synthetic correctional withdrawal-monitoring dataset generator.

ALL DATA PRODUCED HERE IS FICTIONAL. Patient identifiers, timelines, scores,
vital signs, medication names and notes are generated from simple templates.
The score trajectories and vital-sign relationships are *software test
patterns*, not clinical models: they do not encode clinical thresholds and do
not represent validated withdrawal courses or outcomes.

Design
------
* 10 scenario templates x 2 withdrawal types x N patients each (default 5)
  = 100 patients, one booking and one monitoring episode per patient.
* Each scenario = a score trajectory (increasing / decreasing / fluctuating /
  stable) + optional documentation defects (missing assessments, missing vitals,
  irregular intervals, medication documentation gaps, conflicting
  documentation, incomplete monitoring).
* Every patient gets its own RNG seeded with (global seed, patient index), so
  the dataset is fully reproducible and changing one patient's template does
  not perturb the others.
* The injected defects are recorded in `synthetic_ground_truth` so tests and
  evaluation can check whether the analytics find them.
* Assessment timing is configurable (`assessment_interval_hours`); the default
  4 h is a generator parameter, not a claimed clinical schedule.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from backend.database.db import connect_writable
from backend.database.schema import CLINICAL_TABLES, SCHEMA_VERSION, create_schema

GENERATOR_VERSION = "1.0"
TS_FMT = "%Y-%m-%dT%H:%M:%S"

SCALE = {"Alcohol": ("CIWA-Ar", 67), "Opioid": ("COWS", 48)}

SCENARIOS: dict[str, str] = {
    "increasing": "Recorded scores rise over the episode; complete documentation.",
    "decreasing": "Recorded scores fall over the episode; complete documentation.",
    "fluctuating": "Recorded scores rise and fall; complete documentation.",
    "stable": "Recorded scores stay within a narrow band; complete documentation.",
    "missing_assessments": "Several expected assessment time points have no recorded assessment.",
    "missing_vitals": "Some vital-sign sets are absent and some individual measurements are NULL.",
    "irregular_intervals": "Assessment intervals vary widely, including one long gap.",
    "medication_documentation_gaps": "Some scheduled doses are not documented; others refused, held, or not administered.",
    "conflicting_documentation": "A nursing note contradicts a recorded score, and a note contradicts a recorded administration.",
    "incomplete_monitoring": "Monitoring records stop partway through the expected 48 h window.",
}

TRAJECTORIES = ("increasing", "decreasing", "fluctuating", "stable")
FIXED_TRAJECTORY = {
    "increasing": "increasing",
    "decreasing": "decreasing",
    "fluctuating": "fluctuating",
    "stable": "stable",
    "missing_assessments": "decreasing",
    "conflicting_documentation": "increasing",
    "incomplete_monitoring": "increasing",
}

MEDICATIONS = {
    "Alcohol": "Fictional Med AW-1 (demo only)",
    "Opioid": "Fictional Med OW-1 (demo only)",
    "supplement": "Fictional Supplement S-1 (demo only)",
}

NOTE_TEMPLATES = {
    "admission": [
        "Admitted to withdrawal monitoring. Initial assessment completed. Patient alert, oriented, and cooperative.",
        "Placed on withdrawal monitoring per facility protocol. Patient oriented and answering questions appropriately.",
        "Withdrawal monitoring initiated. Patient calm and cooperative during intake assessment.",
    ],
    ("Alcohol", "low"): [
        "Patient resting quietly. Denies nausea. No visible tremor observed at this time. Oral fluids encouraged.",
        "Patient calm and conversant. Reports sleeping intermittently. No sweating noted.",
    ],
    ("Alcohol", "mid"): [
        "Patient reports mild anxiety and intermittent sweating. Fine hand tremor noted on extension. Tolerating oral fluids.",
        "Patient restless at times, reports headache and mild nausea. Slight tremor observed.",
    ],
    ("Alcohol", "high"): [
        "Patient restless and diaphoretic. Reports nausea and feeling anxious. Tremor visible at rest. Findings reported to provider.",
        "Patient agitated, pacing in cell, reports sweating and nausea. Marked tremor observed. Findings reported to provider.",
    ],
    ("Opioid", "low"): [
        "Patient resting. Denies muscle aches. No yawning or runny nose observed.",
        "Patient calm, eating meals. Reports mild difficulty sleeping.",
    ],
    ("Opioid", "mid"): [
        "Patient reports body aches and restlessness. Yawning and runny nose observed. Encouraged fluids.",
        "Patient reports chills and mild stomach upset. Frequent yawning noted.",
    ],
    ("Opioid", "high"): [
        "Patient restless, reports abdominal cramping and diarrhea. Gooseflesh and tearing observed. Findings reported to provider.",
        "Patient unable to sit still, reports vomiting and muscle aches. Pupils appear enlarged. Findings reported to provider.",
    ],
}

CONFLICT_NO_SIGNS_NOTE = "Patient resting comfortably. No signs of withdrawal observed."
CONFLICT_REFUSED_NOTE = "Patient refused scheduled dose of {med}; dose not given."


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 20260311
    patients_per_scenario_per_type: int = 5
    n_assessments: int = 12
    assessment_interval_hours: float = 4.0
    timing_jitter_minutes: float = 15.0
    monitoring_hours: float = 48.0
    admission_window_start: str = "2026-03-01T00:00:00"
    admission_window_days: int = 30


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class Ref:
    """Reference to a generated row, resolved to its ID when serialized."""

    def __init__(self, row: dict, id_key: str):
        self.row, self.id_key = row, id_key


def _ts(dt: datetime) -> str:
    return dt.strftime(TS_FMT)


def _minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def _trajectory_scores(kind: str, t: np.ndarray, rng: np.random.Generator, max_score: int) -> tuple[np.ndarray, dict]:
    """Underlying (pre-noise) score curve on normalized time t in [0, 1]."""
    if kind == "increasing":
        s0, s1 = rng.uniform(2, 7), rng.uniform(16, 28)
        curve = s0 + (s1 - s0) * t
        params = {"start": s0, "end": s1}
        noise = 1.5
    elif kind == "decreasing":
        s0, s1 = rng.uniform(14, 24), rng.uniform(1, 5)
        curve = s1 + (s0 - s1) * np.exp(-3 * t)
        params = {"start": s0, "end": s1}
        noise = 1.5
    elif kind == "fluctuating":
        mid, amp, cycles, phase = rng.uniform(8, 13), rng.uniform(4, 7), rng.uniform(1.5, 3), rng.uniform(0, 2 * math.pi)
        curve = mid + amp * np.sin(2 * math.pi * cycles * t + phase)
        params = {"mid": mid, "amplitude": amp, "cycles": cycles}
        noise = 1.5
    elif kind == "stable":
        level = rng.uniform(3, 8)
        curve = np.full_like(t, level)
        params = {"level": level}
        noise = 1.2
    else:
        raise ValueError(kind)
    scores = np.clip(np.rint(curve + rng.normal(0, noise, size=t.shape)), 0, max_score).astype(int)
    return scores, {k: round(float(v), 2) for k, v in params.items()}


def _relative_level(score: int, scores: list[int]) -> str:
    """Bin a score relative to this patient's own recorded range (no clinical thresholds)."""
    lo, hi = min(scores), max(scores)
    if hi - lo < 4:
        return "low" if hi < 10 else "mid"
    r = (score - lo) / (hi - lo)
    return "low" if r < 0.34 else "mid" if r < 0.67 else "high"


def _vitals(score: float, base: dict, rng: np.random.Generator) -> dict:
    rel = min(max(score / 30.0, 0.0), 1.2)
    sbp = int(round(base["sbp"] + 25 * rel + rng.normal(0, 6)))
    dbp = int(round(base["dbp"] + 12 * rel + rng.normal(0, 4)))
    dbp = min(dbp, sbp - 20)
    return {
        "SBP": sbp,
        "DBP": dbp,
        "Pulse": int(round(base["pulse"] + 30 * rel + rng.normal(0, 4))),
        "Temperature": round(float(base["temp"] + 1.2 * rel + rng.normal(0, 0.25)), 1),
        "RespiratoryRate": int(round(base["rr"] + 4 * rel + rng.normal(0, 1))),
        "OxygenSaturation": int(min(100, round(base["spo2"] - rel + rng.normal(0, 0.6)))),
    }


# --------------------------------------------------------------------------- #
# Per-patient generation
# --------------------------------------------------------------------------- #


@dataclass
class PatientData:
    booking: dict
    assessments: list[dict] = field(default_factory=list)
    vitals: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    administrations: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    ground_truth: dict = field(default_factory=dict)


def generate_patient(index: int, withdrawal_type: str, scenario: str, cfg: GeneratorConfig) -> PatientData:
    rng = np.random.default_rng([cfg.seed, index])
    pid, bkg = f"P{index + 1:03d}", f"B{1001 + index}"
    key = {"PatientID": pid, "BookingNumber": bkg}
    atype, max_score = SCALE[withdrawal_type]

    window0 = datetime.strptime(cfg.admission_window_start, TS_FMT)
    admission = _minute(window0 + timedelta(minutes=float(rng.uniform(0, cfg.admission_window_days * 24 * 60))))
    trajectory = FIXED_TRAJECTORY.get(scenario) or str(rng.choice(TRAJECTORIES))
    details: dict = {"description": SCENARIOS[scenario]}

    # ---- assessment time points -------------------------------------------
    first = float(rng.uniform(0.5, 1.5))
    n = cfg.n_assessments
    if scenario == "irregular_intervals":
        gaps = rng.uniform(1.5, 6.5, size=n - 1)
        long_idx = int(rng.integers(2, n - 2))
        gaps[long_idx] = rng.uniform(9, 12)
        offsets = first + np.concatenate([[0.0], np.cumsum(gaps)])
        offsets = offsets[offsets <= cfg.monitoring_hours]
        details["long_gap_hours"] = round(float(gaps[long_idx]), 2)
        details["long_gap_after_assessment_index"] = long_idx
    else:
        jitter = rng.uniform(-cfg.timing_jitter_minutes, cfg.timing_jitter_minutes, size=n) / 60.0
        jitter[0] = 0.0
        offsets = first + np.arange(n) * cfg.assessment_interval_hours + jitter
    times = [_minute(admission + timedelta(hours=float(h))) for h in offsets]
    tnorm = (offsets - offsets[0]) / max(offsets[-1] - offsets[0], 1e-9)
    true_scores, traj_params = _trajectory_scores(trajectory, tnorm, rng, max_score)
    details["trajectory_params"] = traj_params

    # ---- monitoring end / release ------------------------------------------
    cutoff: datetime | None = None
    release: datetime | None = None
    if scenario == "incomplete_monitoring":
        cutoff_h = float(rng.uniform(16, 24))
        cutoff = admission + timedelta(hours=cutoff_h)
        released = bool(index % 2 == 0)
        if released:
            release = _minute(cutoff + timedelta(hours=float(rng.uniform(0.5, 2))))
        details.update({"records_end_after_hours": round(cutoff_h, 2), "released_early": released})
    elif rng.random() < 0.4:
        release = _minute(admission + timedelta(hours=float(rng.uniform(72, 240))))

    keep = [cutoff is None or t <= cutoff for t in times]

    # ---- assessments --------------------------------------------------------
    dropped_idx: set[int] = set()
    if scenario == "missing_assessments":
        dropped_idx = set(int(i) for i in rng.choice(np.arange(1, len(times) - 1), size=3, replace=False))
        details["expected_but_unrecorded_times"] = sorted(_ts(times[i]) for i in dropped_idx)
    assessments = [
        {**key, "AssessmentDateTime": _ts(t), "AssessmentType": atype, "Score": int(s)}
        for i, (t, s, k) in enumerate(zip(times, true_scores, keep))
        if k and i not in dropped_idx
    ]

    # ---- vital signs (taken around each time point, even if the assessment is missing)
    base = {
        "sbp": rng.uniform(112, 130), "dbp": rng.uniform(70, 82), "pulse": rng.uniform(68, 84),
        "temp": rng.uniform(97.6, 98.6), "rr": rng.uniform(14, 17), "spo2": rng.uniform(96, 99),
    }
    vitals = []
    for t, s, k in zip(times, true_scores, keep):
        if not k:
            continue
        vt = _minute(max(admission + timedelta(minutes=5), t + timedelta(minutes=float(rng.uniform(-10, 10)))))
        vitals.append({**key, "RecordedDateTime": _ts(vt), **_vitals(float(s), base, rng)})
    if scenario == "missing_vitals":
        drop = sorted(int(i) for i in rng.choice(np.arange(1, len(vitals)), size=3, replace=False))
        details["unrecorded_vital_sets_near"] = [vitals[i]["RecordedDateTime"] for i in drop]
        vitals = [v for i, v in enumerate(vitals) if i not in drop]
        idx = [int(i) for i in rng.choice(len(vitals), size=4, replace=False)]
        nulls = {"Temperature": idx[:2], "OxygenSaturation": idx[2:3], "SBP+DBP": idx[3:4]}
        details["null_measurements"] = []
        for fld, rows in nulls.items():
            for r in rows:
                for col in fld.split("+"):
                    vitals[r][col] = None
                details["null_measurements"].append({"vital": Ref(vitals[r], "VitalID"), "fields": fld.split("+")})

    # ---- medication orders & administrations ------------------------------------
    orders, administrations = [], []
    for med, interval_h, start_range in (
        (MEDICATIONS[withdrawal_type], 8, (0.75, 2.0)),
        (MEDICATIONS["supplement"], 24, (1.0, 3.0)),
    ):
        start = _minute(admission + timedelta(hours=float(rng.uniform(*start_range))))
        start = start.replace(minute=start.minute - start.minute % 5)
        planned_stop = start + timedelta(hours=cfg.monitoring_hours)
        status, stop = "Completed", planned_stop
        if release is not None and release < planned_stop:
            status, stop = "Discontinued", release
        elif scenario == "incomplete_monitoring":
            status = "Active"
        order = {**key, "MedicationName": med, "OrderStartDateTime": _ts(start),
                 "OrderStopDateTime": _ts(stop), "OrderStatus": status}
        orders.append(order)
        sched = start
        while sched < stop:
            admin_time = _minute(sched + timedelta(minutes=float(rng.uniform(0, 40))))
            if release is not None:
                admin_time = min(admin_time, release)
            row = {**key, "_order": order, "ScheduledDateTime": _ts(sched),
                   "AdministrationDateTime": _ts(admin_time), "AdministrationStatus": "Administered"}
            if cutoff is not None and sched > cutoff:
                row["AdministrationDateTime"], row["AdministrationStatus"] = None, "Not documented"
            administrations.append(row)
            sched += timedelta(hours=interval_h)

    if scenario == "medication_documentation_gaps":
        pick = [int(i) for i in rng.choice(len(administrations), size=6, replace=False)]
        statuses = ["Not documented"] * 3 + ["Refused", "Held", "Not administered"]
        details["medication_status_changes"] = []
        for i, st in zip(pick, statuses):
            administrations[i]["AdministrationStatus"] = st
            if st == "Not documented":
                administrations[i]["AdministrationDateTime"] = None
            details["medication_status_changes"].append({"administration": Ref(administrations[i], "AdministrationID"), "status": st})
    if scenario == "incomplete_monitoring":
        undocumented = [Ref(a, "AdministrationID") for a in administrations if a["AdministrationStatus"] == "Not documented"]
        details["undocumented_after_records_end"] = undocumented

    # ---- nursing notes -------------------------------------------------------
    recorded_scores = [a["Score"] for a in assessments]
    notes: list[dict] = []
    note_time = _minute(admission + timedelta(hours=float(rng.uniform(0.3, 1.0))))
    notes.append({**key, "NoteDateTime": _ts(note_time), "NoteText": str(rng.choice(NOTE_TEMPLATES["admission"]))})
    for frac in (0.5, 1.0):
        a = assessments[min(int(frac * (len(assessments) - 1)), len(assessments) - 1)]
        t = datetime.strptime(a["AssessmentDateTime"], TS_FMT) + timedelta(minutes=float(rng.uniform(5, 30)))
        level = _relative_level(a["Score"], recorded_scores)
        notes.append({**key, "NoteDateTime": _ts(_minute(t)),
                      "NoteText": str(rng.choice(NOTE_TEMPLATES[(withdrawal_type, level)]))})

    if scenario == "conflicting_documentation":
        top = max(assessments, key=lambda a: (a["Score"], a["AssessmentDateTime"]))
        t = datetime.strptime(top["AssessmentDateTime"], TS_FMT) + timedelta(minutes=10)
        n1 = {**key, "NoteDateTime": _ts(t), "NoteText": CONFLICT_NO_SIGNS_NOTE}
        dose = [a for a in administrations if a["_order"] is orders[0]][2]
        t2 = datetime.strptime(dose["AdministrationDateTime"], TS_FMT) + timedelta(minutes=15)
        n2 = {**key, "NoteDateTime": _ts(t2), "NoteText": CONFLICT_REFUSED_NOTE.format(med=orders[0]["MedicationName"])}
        notes += [n1, n2]
        details["conflicts"] = [
            {"type": "note_contradicts_assessment", "note": Ref(n1, "NoteID"), "assessment": Ref(top, "AssessmentID")},
            {"type": "note_contradicts_administration", "note": Ref(n2, "NoteID"), "administration": Ref(dose, "AdministrationID")},
        ]

    booking = {**key, "AdmissionDateTime": _ts(admission),
               "ReleaseDateTime": _ts(release) if release else None, "WithdrawalType": withdrawal_type}
    return PatientData(
        booking=booking,
        assessments=assessments,
        vitals=vitals,
        orders=orders,
        administrations=administrations,
        notes=sorted(notes, key=lambda r: r["NoteDateTime"]),
        ground_truth={**key, "Scenario": scenario, "Trajectory": trajectory, "_details": details},
    )


# --------------------------------------------------------------------------- #
# Whole dataset
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    config: GeneratorConfig
    patients: list[PatientData]

    def tables(self) -> dict[str, list[dict]]:
        p = self.patients
        return {
            "patient_bookings": [x.booking for x in p],
            "withdrawal_assessments": [r for x in p for r in x.assessments],
            "vital_signs": [r for x in p for r in x.vitals],
            "medication_orders": [r for x in p for r in x.orders],
            "medication_administrations": [r for x in p for r in x.administrations],
            "nursing_notes": [r for x in p for r in x.notes],
            "synthetic_ground_truth": [x.ground_truth for x in p],
        }


def _assign_ids(patients: list[PatientData]) -> None:
    """Sequential IDs in patient order, chronological within each patient."""
    counters = {"ASM": 0, "VIT": 0, "ORD": 0, "MAR": 0, "NOTE": 0}

    def nxt(prefix: str, width: int) -> str:
        counters[prefix] += 1
        return f"{prefix}-{counters[prefix]:0{width}d}"

    for p in patients:
        for r in sorted(p.assessments, key=lambda r: r["AssessmentDateTime"]):
            r["AssessmentID"] = nxt("ASM", 5)
        for r in sorted(p.vitals, key=lambda r: r["RecordedDateTime"]):
            r["VitalID"] = nxt("VIT", 5)
        for r in p.orders:
            r["OrderID"] = nxt("ORD", 4)
        for r in sorted(p.administrations, key=lambda r: (r["ScheduledDateTime"], r["_order"]["OrderID"])):
            r["AdministrationID"] = nxt("MAR", 5)
            r["OrderID"] = r["_order"]["OrderID"]
        for r in p.notes:
            r["NoteID"] = nxt("NOTE", 4)
        p.assessments.sort(key=lambda r: r["AssessmentID"])
        p.vitals.sort(key=lambda r: r["VitalID"])
        p.administrations.sort(key=lambda r: r["AdministrationID"])
        details = p.ground_truth.pop("_details")
        p.ground_truth["Details"] = json.dumps(
            details, sort_keys=True, default=lambda o: o.row[o.id_key] if isinstance(o, Ref) else str(o)
        )
    for p in patients:
        for r in p.administrations:
            r.pop("_order", None)


def generate_dataset(cfg: GeneratorConfig | None = None) -> Dataset:
    cfg = cfg or GeneratorConfig()
    plan = [
        (wtype, scenario)
        for wtype in ("Alcohol", "Opioid")
        for scenario in SCENARIOS
        for _ in range(cfg.patients_per_scenario_per_type)
    ]
    # Shuffle so patient numbers don't reveal scenario or withdrawal type.
    order = np.random.default_rng(cfg.seed).permutation(len(plan))
    patients = [generate_patient(i, *plan[int(j)], cfg) for i, j in enumerate(order)]
    _assign_ids(patients)
    return Dataset(cfg, patients)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #

COLUMNS = {
    "patient_bookings": ["PatientID", "BookingNumber", "AdmissionDateTime", "ReleaseDateTime", "WithdrawalType"],
    "withdrawal_assessments": ["AssessmentID", "PatientID", "BookingNumber", "AssessmentDateTime", "AssessmentType", "Score"],
    "vital_signs": ["VitalID", "PatientID", "BookingNumber", "RecordedDateTime", "SBP", "DBP", "Pulse",
                    "Temperature", "RespiratoryRate", "OxygenSaturation"],
    "medication_orders": ["OrderID", "PatientID", "BookingNumber", "MedicationName", "OrderStartDateTime",
                          "OrderStopDateTime", "OrderStatus"],
    "medication_administrations": ["AdministrationID", "OrderID", "PatientID", "BookingNumber", "ScheduledDateTime",
                                   "AdministrationDateTime", "AdministrationStatus"],
    "nursing_notes": ["NoteID", "PatientID", "BookingNumber", "NoteDateTime", "NoteText"],
    "synthetic_ground_truth": ["PatientID", "BookingNumber", "Scenario", "Trajectory", "Details"],
}


def write_sqlite(ds: Dataset, path: Path, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass overwrite=True (--overwrite) to replace it.")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    conn = connect_writable(tmp)
    try:
        create_schema(conn)
        with conn:
            for table, rows in ds.tables().items():
                cols = COLUMNS[table]
                conn.executemany(
                    f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [tuple(r[c] for c in cols) for r in rows],
                )
            info = {
                "notice": "SYNTHETIC DATA. All patients and records are fictional, for software demonstration only.",
                "generator_version": GENERATOR_VERSION,
                "schema_version": SCHEMA_VERSION,
                "seed": str(ds.config.seed),
                "config": json.dumps(asdict(ds.config), sort_keys=True),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            conn.executemany("INSERT INTO dataset_info VALUES (?, ?)", list(info.items()))
    finally:
        conn.close()
    os.replace(tmp, path)


def export_csv(ds: Dataset, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for table, rows in ds.tables().items():
        p = out_dir / f"{table}.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS[table], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        written.append(p)
    return written


__all__ = ["CLINICAL_TABLES", "GeneratorConfig", "SCENARIOS", "generate_dataset", "write_sqlite", "export_csv"]
