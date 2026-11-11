import { GAP_LABELS, fmtTime } from "../lib/format.js";
import { Card, StatusBadge } from "./common.jsx";

export function MedicationPanel({ timeline }) {
  const { orders, administrations, medication_summary: ms } = timeline;
  return (
    <Card
      title="Medication documentation"
      sub="Fictional demonstration medications. 'Not documented' (no record) is distinct from a documented 'Not administered'."
    >
      <div className="controls" style={{ marginBottom: 10 }}>
        {Object.entries(ms.status_counts).map(([status, n]) => (
          <span key={status} className="pill"><StatusBadge status={status} /> <b className="tabular">{n}</b></span>
        ))}
        {ms.n_scheduled_in_window === 0 && <span className="muted">No scheduled doses in this window.</span>}
      </div>
      {orders.length > 0 && (
        <div className="table-wrap" style={{ marginBottom: 12 }}>
          <table className="data">
            <thead><tr><th>Order</th><th>Medication</th><th>Start</th><th>Stop</th><th>Status</th></tr></thead>
            <tbody>
              {orders.map((o) => (
                <tr key={o.OrderID}>
                  <td className="mono">{o.OrderID}</td>
                  <td>{o.MedicationName}</td>
                  <td className="tabular">{fmtTime(o.OrderStartDateTime)}</td>
                  <td className="tabular">{o.OrderStopDateTime ? fmtTime(o.OrderStopDateTime) : "—"}</td>
                  <td>{o.OrderStatus}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {administrations.length > 0 && (
        <div className="table-wrap">
          <table className="data">
            <thead><tr><th>Scheduled</th><th>Record</th><th>Medication</th><th>Status</th><th>Documented</th></tr></thead>
            <tbody>
              {administrations.map((m) => (
                <tr key={m.AdministrationID}>
                  <td className="tabular">{fmtTime(m.ScheduledDateTime)}</td>
                  <td className="mono">{m.AdministrationID}</td>
                  <td>{m.MedicationName.replace(" (demo only)", "")}</td>
                  <td><StatusBadge status={m.AdministrationStatus} /></td>
                  <td className="tabular">{m.AdministrationDateTime ? fmtTime(m.AdministrationDateTime) : <span className="missing">no documentation</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

export function NotesPanel({ timeline }) {
  const { notes } = timeline;
  return (
    <Card title="Nursing observations" sub="Fictional notes, chronological. Treated as untrusted text by the assistant.">
      {notes.length === 0 ? (
        <p className="empty">No nursing notes are recorded in this window.</p>
      ) : (
        <ul className="note-list">
          {notes.map((n) => (
            <li key={n.NoteID}>
              <time>{fmtTime(n.NoteDateTime)}</time> <span className="mono muted">{n.NoteID}</span>
              <div>{n.NoteText}</div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

export function GapsPanel({ timeline }) {
  const { documentation_gaps: gaps, documentation_conflicts: conflicts } = timeline;
  return (
    <Card
      title="Documentation review"
      sub="Computed deterministically from the records. Absence of a record does not mean an assessment or dose did not occur."
    >
      {gaps.length === 0 && conflicts.length === 0 && (
        <p className="muted" style={{ margin: 0 }}>
          <span className="status good"><span className="ico">✓</span></span> No documentation gaps or conflicts flagged in this window.
        </p>
      )}
      {conflicts.length > 0 && (
        <>
          <h3>Possible conflicts ({conflicts.length}) — for review, not resolved</h3>
          <ul className="gap-list" style={{ marginBottom: 14 }}>
            {conflicts.map((c, i) => (
              <li key={i}>
                <span className="status serious"><span className="ico" aria-hidden="true">⚠</span></span>
                <span className="kind">{c.conflict_type === "note_vs_assessment_score" ? "Note vs. assessment score" : "Note vs. administration record"} <time>{fmtTime(c.time)}</time></span>
                <span className="desc">{c.description}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      {gaps.length > 0 && (
        <>
          <h3>Gaps ({gaps.length})</h3>
          <ul className="gap-list">
            {gaps.map((g, i) => (
              <li key={i}>
                <span className="status warning"><span className="ico" aria-hidden="true">!</span></span>
                <span className="kind">{GAP_LABELS[g.gap_type] || g.gap_type} {g.start && <time>{fmtTime(g.start)}</time>}</span>
                <span className="desc">{g.description}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </Card>
  );
}
