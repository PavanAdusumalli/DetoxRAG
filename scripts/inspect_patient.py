"""Print one synthetic patient's raw records in chronological order.

Usage:
    python scripts/inspect_patient.py P001
    python scripts/inspect_patient.py P001 --ground-truth     # also show injected scenario/defects
    python scripts/inspect_patient.py --list                  # one line per patient
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database.db import connect_readonly  # noqa: E402


def show_list(conn) -> None:
    rows = conn.execute("""
        SELECT b.PatientID, b.BookingNumber, b.WithdrawalType, b.AdmissionDateTime, b.ReleaseDateTime,
               g.Scenario, g.Trajectory,
               (SELECT COUNT(*) FROM withdrawal_assessments a WHERE a.BookingNumber = b.BookingNumber) AS n_assess
        FROM patient_bookings b JOIN synthetic_ground_truth g USING (BookingNumber)
        ORDER BY b.PatientID""").fetchall()
    for r in rows:
        print(f"{r['PatientID']} {r['BookingNumber']} {r['WithdrawalType']:7s} adm {r['AdmissionDateTime']} "
              f"rel {r['ReleaseDateTime'] or '-':19s} {r['n_assess']:2d} assessments  "
              f"[{r['Scenario']} / {r['Trajectory']}]")


def show_patient(conn, pid: str, ground_truth: bool) -> int:
    b = conn.execute("SELECT * FROM patient_bookings WHERE PatientID = ?", (pid,)).fetchone()
    if b is None:
        print(f"No patient {pid!r}")
        return 1
    bk = b["BookingNumber"]
    print(f"SYNTHETIC PATIENT {b['PatientID']}  booking {bk}  {b['WithdrawalType']} withdrawal monitoring")
    print(f"Admitted {b['AdmissionDateTime']}   Released {b['ReleaseDateTime'] or '(still in custody)'}\n")

    events = []
    for r in conn.execute("SELECT * FROM withdrawal_assessments WHERE BookingNumber = ?", (bk,)):
        events.append((r["AssessmentDateTime"], r["AssessmentID"], f"{r['AssessmentType']} score {r['Score']}"))
    for r in conn.execute("SELECT * FROM vital_signs WHERE BookingNumber = ?", (bk,)):
        def f(v, u=""):
            return "—" if v is None else f"{v}{u}"
        events.append((r["RecordedDateTime"], r["VitalID"],
                       f"BP {f(r['SBP'])}/{f(r['DBP'])}  P {f(r['Pulse'])}  T {f(r['Temperature'], '°F')}  "
                       f"RR {f(r['RespiratoryRate'])}  SpO2 {f(r['OxygenSaturation'], '%')}"))
    for r in conn.execute("""SELECT m.*, o.MedicationName FROM medication_administrations m
                             JOIN medication_orders o USING (OrderID) WHERE m.BookingNumber = ?""", (bk,)):
        when = r["AdministrationDateTime"] or "no documentation"
        events.append((r["ScheduledDateTime"], r["AdministrationID"],
                       f"{r['MedicationName']} scheduled → {r['AdministrationStatus']} ({when})"))
    for r in conn.execute("SELECT * FROM nursing_notes WHERE BookingNumber = ?", (bk,)):
        events.append((r["NoteDateTime"], r["NoteID"], f"NOTE: {r['NoteText']}"))

    print("Orders:")
    for o in conn.execute("SELECT * FROM medication_orders WHERE BookingNumber = ?", (bk,)):
        print(f"  {o['OrderID']} {o['MedicationName']}  {o['OrderStartDateTime']} → {o['OrderStopDateTime']}  [{o['OrderStatus']}]")
    print("\nTimeline:")
    for ts, rid, text in sorted(events):
        print(f"  {ts.replace('T', ' ')}  {rid:10s} {text}")

    if ground_truth:
        g = conn.execute("SELECT * FROM synthetic_ground_truth WHERE BookingNumber = ?", (bk,)).fetchone()
        print(f"\nGROUND TRUTH (generator only): scenario={g['Scenario']} trajectory={g['Trajectory']}")
        print(json.dumps(json.loads(g["Details"]), indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patient_id", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ground-truth", action="store_true")
    args = ap.parse_args()
    conn = connect_readonly()
    if args.list or not args.patient_id:
        show_list(conn)
        return 0
    return show_patient(conn, args.patient_id.upper(), args.ground_truth)


if __name__ == "__main__":
    raise SystemExit(main())
