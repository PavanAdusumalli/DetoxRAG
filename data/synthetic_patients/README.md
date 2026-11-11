# Synthetic withdrawal-monitoring dataset

> **ALL RECORDS ARE FICTIONAL.** No real patient information was used. Score trajectories, vital-sign relationships, medication names and nursing notes are software test patterns generated from templates. They do not encode clinical thresholds and do not represent validated withdrawal courses, treatments, or outcomes.

## Regenerate

```bash
python scripts/generate_synthetic_data.py --overwrite --csv
python scripts/inspect_patient.py --list
python scripts/inspect_patient.py P015 --ground-truth
```

`detoxrag.db` is not committed (it is rebuilt deterministically from the seed). The `csv/` export is committed for browsing.

## Contents (default seed 20260311)

| Table | Rows | Notes |
|---|---:|---|
| `patient_bookings` | 100 | 50 Alcohol, 50 Opioid; one booking per patient. IDs `P001`–`P100`, `B1001`–`B1100`. |
| `withdrawal_assessments` | 1,097 | CIWA-Ar (0–67) for Alcohol, COWS (0–48) for Opioid. 12 per patient unless a scenario removes some. |
| `vital_signs` | 1,097 | Units in `measurement_units`: mmHg, beats/min, °F, breaths/min, %. NULL = not recorded. |
| `medication_orders` | 200 | Two explicitly fictional orders per booking; no doses or routes. |
| `medication_administrations` | 775 | Statuses: Administered, Not administered, Refused, Held, Not documented. |
| `nursing_notes` | 320 | Templated observations aligned with the patient's own score trajectory. |
| **Total** | **3,589** | Below the ~3,800 target because missing-data scenarios intentionally remove records. |

Support tables: `dataset_info` (seed, config, notice), `measurement_units`, and `synthetic_ground_truth` (see below).

## Scenarios

10 templates × 2 withdrawal types × 5 patients. Patient numbers are shuffled so they don't reveal the scenario.

| Scenario | Score trajectory | Injected documentation pattern |
|---|---|---|
| `increasing` | rising | none |
| `decreasing` | falling (exponential-like decay) | none |
| `fluctuating` | sinusoidal, 1.5–3 cycles | none |
| `stable` | flat, small noise | none |
| `missing_assessments` | falling | 3 interior time points have no assessment row (vitals still recorded) |
| `missing_vitals` | random | 3 vital-sign sets absent; 2 NULL temperatures, 1 NULL SpO2, 1 NULL BP |
| `irregular_intervals` | random | intervals 1.5–6.5 h plus one 9–12 h gap |
| `medication_documentation_gaps` | random | 3 doses Not documented, 1 Refused, 1 Held, 1 Not administered |
| `conflicting_documentation` | rising | a "no signs of withdrawal" note 10 min after the highest score; a "patient refused" note after a dose recorded as Administered |
| `incomplete_monitoring` | rising | records stop after 16–24 h; half released early, half remain in custody with later doses Not documented |

## Simulation assumptions

- **Timing.** First assessment 0.5–1.5 h after admission, then every `assessment_interval_hours` (default 4 h, configurable via `--interval-hours`) with ±15 min jitter. This default is a generator parameter, not a claimed clinical schedule.
- **Scores.** Trajectory curve + Gaussian noise (sd 1.2–1.5), rounded and clipped to the instrument range. The same numeric ranges are used for both instruments, but the instruments are never mixed or compared.
- **Vital signs.** Per-patient random baseline plus a small linear dependence on the current score, plus noise. This correlation exists only so charts look coherent; it is not a physiological model. CHECK constraints are data-plausibility bounds to catch bugs, not clinical limits.
- **Medications.** "Fictional Med AW-1 / OW-1" every 8 h and "Fictional Supplement S-1" every 24 h for 48 h. Names are deliberately not real drugs, and no doses are stored, so nothing in the dataset can be read as a treatment recommendation.
- **Notes.** Templates chosen by the score's position within *that patient's own* recorded range (low/mid/high third), not by any clinical cut-off.
- **Release.** 40% of non-incomplete bookings have a release 3–10 days after admission (after monitoring); the rest are still in custody.
- **Missing vs. not done.** A missing assessment is an absent row, never a NULL score: the dataset shows only what was *recorded*. `Not documented` (no record of a scheduled dose) is distinct from a documented `Not administered`.

## Ground truth table

`synthetic_ground_truth` records each booking's scenario, trajectory parameters and the exact injected defects (e.g. which `NoteID` contradicts which `AssessmentID`). It exists so tests and the Phase 8 evaluation can check whether the analytics detect these patterns. **It is not part of the simulated health record**; application retrieval code must not read it or pass it to the language model.

## Integrity guarantees

Enforced by the schema: composite foreign keys (a record can't attach to another patient's booking), instrument–withdrawal-type trigger, per-instrument score ranges, ISO-8601 timestamp format, SBP > DBP, `Not documented` ⇔ no administration time. Checked by `backend/database/validation.py` (run automatically after generation): records within admission/release window, doses within their order window, chronological IDs, every booking has assessments.
