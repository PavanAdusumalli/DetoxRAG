import AssessmentChart from "../components/AssessmentChart.jsx";
import { Card, ErrorBox, Loading, Tile, TypePill } from "../components/common.jsx";
import { GapsPanel, MedicationPanel, NotesPanel } from "../components/RecordPanels.jsx";
import VitalsCharts from "../components/VitalsCharts.jsx";
import { useApi } from "../hooks/useApi.js";
import { fmtHours, fmtSigned, fmtTime } from "../lib/format.js";
import { api } from "../services/api.js";

export const WINDOWS = [
  { label: "Entire booking", hours: null },
  { label: "Last 48 h", hours: 48 },
  { label: "Last 24 h", hours: 24 },
  { label: "Last 12 h", hours: 12 },
  { label: "Last 6 h", hours: 6 },
];

export function WindowSelector({ hours, onChange }) {
  return (
    <div className="seg" role="group" aria-label="Time window" style={{ minWidth: 0, flexWrap: "wrap" }}>
      {WINDOWS.map((w) => (
        <button key={w.label} aria-pressed={hours === w.hours} onClick={() => onChange(w.hours)}>
          {w.label}
        </button>
      ))}
    </div>
  );
}

export function PatientHeader({ timeline, hours, onHours, extra }) {
  const { booking: b, window: w, score_summary: s } = timeline;
  return (
    <Card>
      <div className="card-head" style={{ marginBottom: 12 }}>
        <div>
          <h2 style={{ fontSize: 18 }}>
            Patient {b.PatientID} <span className="muted" style={{ fontWeight: 400 }}>· Booking {b.BookingNumber}</span>{" "}
            <TypePill type={b.WithdrawalType} />
          </h2>
          <p className="sub" style={{ margin: 0 }}>
            {w.label}: {fmtTime(w.start)} → {fmtTime(w.end)} ({fmtHours(w.hours)}). Window end = {w.end_reference}.
          </p>
        </div>
        <div className="controls">
          <WindowSelector hours={hours} onChange={onHours} />
          {extra}
        </div>
      </div>
      <div className="tiles">
        <Tile label="Monitoring scale" value={s.assessment_type} note={`range 0–${s.scale_max}`} />
        <Tile label="Admitted" value={fmtTime(b.AdmissionDateTime)} note={b.ReleaseDateTime ? `Released ${fmtTime(b.ReleaseDateTime)}` : "Still in custody"} />
        <Tile label="Assessments in window" value={s.n_assessments} note={s.n_assessments > 1 ? `over ${fmtHours(s.observation_span_hours)}` : null} />
        <Tile label="Latest recorded score" value={s.latest ? s.latest.score : "—"} note={s.latest ? fmtTime(s.latest.time) : "none recorded"} />
        <Tile label="Change (earliest → latest)" value={fmtSigned(s.change)} note={s.earliest ? `${s.earliest.score} → ${s.latest.score} · ${s.net_direction}` : s.net_direction} />
        <Tile label="Flags" value={timeline.documentation_gaps.length + timeline.documentation_conflicts.length}
          note={`${timeline.documentation_gaps.length} gaps · ${timeline.documentation_conflicts.length} conflicts`} />
      </div>
    </Card>
  );
}

export default function Dashboard({ patientId, hours, onHours, onAsk }) {
  const tl = useApi(() => (patientId ? api.timeline(patientId, hours) : Promise.resolve(null)), [patientId, hours]);
  if (!patientId) return <Card><p className="empty">Select a synthetic patient from the list.</p></Card>;
  if (tl.error) return <ErrorBox error={tl.error} onRetry={tl.reload} />;
  if (!tl.data) return <Card><Loading label="Loading patient timeline…" /></Card>;
  const t = tl.data;
  return (
    <div className="stack">
      <PatientHeader timeline={t} hours={hours} onHours={onHours}
        extra={<button className="btn primary" onClick={onAsk}>Ask the assistant →</button>} />
      <AssessmentChart timeline={t} />
      <div className="grid grid-2">
        <GapsPanel timeline={t} />
        <NotesPanel timeline={t} />
      </div>
      <VitalsCharts timeline={t} />
      <MedicationPanel timeline={t} />
    </div>
  );
}
