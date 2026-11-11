"""Create the SQLite database and fill it with the synthetic patient dataset.

Usage:
    python scripts/generate_synthetic_data.py                 # create (refuses to overwrite)
    python scripts/generate_synthetic_data.py --overwrite     # rebuild from scratch
    python scripts/generate_synthetic_data.py --overwrite --csv   # also export CSVs
    python scripts/generate_synthetic_data.py --seed 7 --interval-hours 6 --out /tmp/alt.db

The same seed and options always produce identical records.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import get_settings  # noqa: E402
from backend.database.db import connect_readonly  # noqa: E402
from backend.database.synthetic_generator import (  # noqa: E402
    GeneratorConfig,
    export_csv,
    generate_dataset,
    write_sqlite,
)
from backend.database.validation import run_integrity_checks, table_counts  # noqa: E402


def main() -> int:
    d = GeneratorConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=get_settings().database_path)
    ap.add_argument("--overwrite", action="store_true", help="Replace an existing database file")
    ap.add_argument("--csv", action="store_true", help="Also export CSV files next to the database")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--per-scenario", type=int, default=d.patients_per_scenario_per_type,
                    help="Patients per scenario per withdrawal type (10 scenarios x 2 types)")
    ap.add_argument("--n-assessments", type=int, default=d.n_assessments)
    ap.add_argument("--interval-hours", type=float, default=d.assessment_interval_hours)
    args = ap.parse_args()

    cfg = GeneratorConfig(
        seed=args.seed,
        patients_per_scenario_per_type=args.per_scenario,
        n_assessments=args.n_assessments,
        assessment_interval_hours=args.interval_hours,
    )
    ds = generate_dataset(cfg)
    try:
        write_sqlite(ds, args.out, overwrite=args.overwrite)
    except FileExistsError as e:
        print(f"[stop] {e}")
        return 1
    print(f"Wrote {args.out}")

    conn = connect_readonly(args.out)
    counts = table_counts(conn)
    total = sum(counts.values())
    print("\nRecords per table:")
    for t, n in counts.items():
        print(f"  {t:28s} {n:6d}")
    print(f"  {'TOTAL':28s} {total:6d}")

    types = Counter(r[0] for r in conn.execute("SELECT WithdrawalType FROM patient_bookings"))
    scen = Counter(r[0] for r in conn.execute("SELECT Scenario FROM synthetic_ground_truth"))
    status = Counter(r[0] for r in conn.execute("SELECT AdministrationStatus FROM medication_administrations"))
    print(f"\nWithdrawal types: {dict(types)}")
    print(f"Scenarios: {dict(scen)}")
    print(f"Administration statuses: {dict(status)}")

    results = run_integrity_checks(conn)
    failed = {k: v for k, v in results.items() if v}
    print("\nIntegrity checks:")
    for name, bad in results.items():
        print(f"  [{'FAIL' if bad else ' ok '}] {name}" + (f": {bad[:5]}" if bad else ""))
    conn.close()

    if args.csv:
        out_dir = args.out.parent / "csv"
        paths = export_csv(ds, out_dir)
        print(f"\nCSV export: {len(paths)} files in {out_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
