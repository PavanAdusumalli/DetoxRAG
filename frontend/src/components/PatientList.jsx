import { useMemo, useState } from "react";
import { fmtTime } from "../lib/format.js";
import { ErrorBox, Loading } from "./common.jsx";

export default function PatientList({ patients, loading, error, reload, selected, onSelect }) {
  const [query, setQuery] = useState("");
  const [type, setType] = useState("All");

  const rows = useMemo(() => {
    const q = query.trim().toUpperCase();
    return (patients || []).filter(
      (p) =>
        (type === "All" || p.WithdrawalType === type) &&
        (!q || p.PatientID.includes(q) || p.BookingNumber.includes(q)),
    );
  }, [patients, query, type]);

  return (
    <aside className="card sidebar" aria-label="Synthetic patients">
      <div className="sidebar-head">
        <input
          className="search"
          placeholder="Search patient or booking…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search patients"
        />
        <div className="seg" role="group" aria-label="Withdrawal type">
          {["All", "Alcohol", "Opioid"].map((t) => (
            <button key={t} aria-pressed={type === t} onClick={() => setType(t)}>
              {t}
            </button>
          ))}
        </div>
        <div className="muted" style={{ fontSize: 12 }}>
          {rows.length} of {patients?.length ?? 0} synthetic patients
        </div>
      </div>
      <div className="sidebar-list">
        {loading && <Loading />}
        <ErrorBox error={error} onRetry={reload} />
        {rows.map((p) => (
          <button
            key={p.BookingNumber}
            className="patient-row"
            aria-current={selected === p.PatientID}
            onClick={() => onSelect(p.PatientID)}
          >
            <span className="pid">
              {p.PatientID} <span className="muted" style={{ fontWeight: 400 }}>· {p.BookingNumber}</span>
            </span>
            <span className="score">
              {p.AssessmentType} {p.latest_score ?? "—"}
            </span>
            <span className="meta">
              {p.WithdrawalType} · admitted {fmtTime(p.AdmissionDateTime)} · {p.n_assessments} assessments
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}
