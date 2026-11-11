import { statusMeta } from "../lib/format.js";

export function StatusBadge({ status }) {
  const m = statusMeta(status);
  return (
    <span className={`status ${m.tone}`}>
      <span className="ico" aria-hidden="true">{m.icon}</span>
      {m.label}
    </span>
  );
}

export function TypePill({ type }) {
  return <span className={`pill ${type === "Alcohol" ? "alcohol" : "opioid"}`}>{type}</span>;
}

export function Loading({ label = "Loading…" }) {
  return (
    <div className="empty" role="status">
      <span className="spinner" /> {label}
    </div>
  );
}

export function ErrorBox({ error, onRetry }) {
  if (!error) return null;
  return (
    <div className="callout error" role="alert">
      <strong>{error.status ? `Error ${error.status}: ` : ""}</strong>
      {error.message}
      {onRetry && (
        <>
          {" "}
          <button className="btn small" onClick={onRetry}>Retry</button>
        </>
      )}
    </div>
  );
}

export function Tile({ label, value, note }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export function Card({ title, sub, actions, children, className = "" }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {sub && <p className="sub">{sub}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}
