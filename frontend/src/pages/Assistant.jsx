import { useEffect, useRef, useState } from "react";
import { Card, ErrorBox, Loading } from "../components/common.jsx";
import { useApi } from "../hooks/useApi.js";
import { fmtMs, fmtSigned, fmtTime, isGuidelineId, splitCitations } from "../lib/format.js";
import { api } from "../services/api.js";
import { PatientHeader } from "./Dashboard.jsx";

const EXAMPLES = [
  "Summarize this patient's withdrawal progression over the last 12 hours.",
  "How have the recorded assessment scores changed?",
  "What vital-sign measurements are documented?",
  "Which relevant guideline passages discuss withdrawal monitoring?",
  "What information is missing from the retrieved record?",
  "Summarize documented medication administrations over this period.",
];

const MODES = [
  { value: "hybrid", label: "Hybrid RAG (dense + BM25)" },
  { value: "dense", label: "Dense RAG only" },
  { value: "bm25", label: "BM25 only" },
  { value: "none", label: "No guidelines (records only)" },
];

function CitedText({ text, active, onCite }) {
  if (!text) return <p className="answer-text muted">—</p>;
  return (
    <p className="answer-text">
      {splitCitations(text).map((part, i) =>
        part.type === "text" ? (
          <span key={i}>{part.value}</span>
        ) : (
          <button key={i} className={`cite ${isGuidelineId(part.value) ? "guideline" : ""}`}
            aria-pressed={active === part.value} onClick={() => onCite(part.value)} title="Show evidence">
            {part.value}
          </button>
        ),
      )}
    </p>
  );
}

function Section({ n, title, children }) {
  return (
    <div className="answer-section">
      <h3><span className="num">{n}</span>{title}</h3>
      {children}
    </div>
  );
}

function EvidencePanel({ result, active, refs }) {
  if (!result) return null;
  const patient = result.patient_evidence || [];
  const guideline = result.guideline_citations || result.citations || [];
  const passages = result.retrieved_passages || result.passages || [];
  return (
    <Card title="Evidence" sub="Every citation resolves to a database record, a computed fact, or a retrieved guideline passage." className="evidence-panel">
      {patient.length > 0 && <h3>Patient-record evidence ({patient.length})</h3>}
      {patient.map((e) => (
        <div key={e.evidence_id} ref={(el) => (refs.current[e.evidence_id] = el)} className={`evidence-item ${active === e.evidence_id ? "active" : ""}`}>
          <div className="id">{e.evidence_id} <span className="meta">· {e.kind}{e.time ? ` · ${fmtTime(e.time)}` : ""}</span></div>
          <blockquote>{e.text}</blockquote>
        </div>
      ))}
      {guideline.length > 0 && <h3 style={{ marginTop: 14 }}>Guideline citations ({guideline.length})</h3>}
      {guideline.map((c) => (
        <div key={c.citation_id} ref={(el) => (refs.current[c.citation_id] = el)} className={`evidence-item ${active === c.citation_id ? "active" : ""}`}>
          <div className="id">{c.citation_id}</div>
          <div className="meta">
            {c.publisher.split("(")[0].trim()} ({c.publication_date}) · {c.section_path} · {c.pages} (PDF p. {c.pdf_page_start})
            {c.recommendation_ids?.length > 0 && ` · recs ${c.recommendation_ids.join(", ")}`}
          </div>
          <blockquote>{c.quote}</blockquote>
          <a href={c.source_url} target="_blank" rel="noreferrer" style={{ fontSize: 12 }}>Source document ↗</a>
        </div>
      ))}
      {passages.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary>All retrieved passages ({passages.length})</summary>
          {passages.map((p) => (
            <div key={p.chunk.chunk_id} className="evidence-item" style={{ marginTop: 8 }}>
              <div className="id">#{p.rank} {p.chunk.chunk_id}
                <span className="meta"> · {p.dense_rank ? `dense #${p.dense_rank}` : ""}{p.dense_rank && p.bm25_rank ? " · " : ""}{p.bm25_rank ? `BM25 #${p.bm25_rank}` : ""}</span>
              </div>
              <div className="meta">{p.chunk.section_path}</div>
              <blockquote>{p.chunk.text}</blockquote>
            </div>
          ))}
        </details>
      )}
      {!patient.length && !guideline.length && !passages.length && <p className="muted">No evidence cited.</p>}
    </Card>
  );
}

function ValidationNotes({ result }) {
  const v = result.validation;
  const warnings = result.warnings || [];
  return (
    <>
      {warnings.length > 0 && (
        <div className="callout warn">
          <strong>Validation notes</strong>
          <ul>{warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </div>
      )}
      {v && warnings.length === 0 && (
        <div className="callout info">
          ✓ All citations verified against supplied evidence; no contradicted values, removed sentences or other-patient identifiers.
        </div>
      )}
    </>
  );
}

function PatientAnswer({ result, active, onCite }) {
  const s = result.score_summary;
  return (
    <Card title="Review summary" sub={`${result.patient_id} · ${result.booking_number} · ${result.monitoring_type} · ${result.time_window.label}`}>
      <div className="stack" style={{ gap: 12 }}>
        <ValidationNotes result={result} />
        <Section n={1} title="Patient summary"><CitedText text={result.patient_summary} active={active} onCite={onCite} /></Section>
        <Section n={2} title="Assessment trend">
          <CitedText text={result.assessment_trend} active={active} onCite={onCite} />
          {s.n_assessments > 0 && (
            <p className="sub" style={{ margin: "6px 0 0" }}>
              Computed: {s.n_assessments} assessments · earliest {s.earliest.score} · latest {s.latest.score} · change {fmtSigned(s.change)} · range {s.minimum.score}–{s.maximum.score}
            </p>
          )}
        </Section>
        <Section n={3} title="Vital-sign observations"><CitedText text={result.vital_sign_observations} active={active} onCite={onCite} /></Section>
        <Section n={4} title="Medication documentation"><CitedText text={result.medication_documentation} active={active} onCite={onCite} /></Section>
        <Section n={5} title={`Missing information (${result.documentation_gaps.length + result.documentation_conflicts.length})`}>
          {result.documentation_gaps.length + result.documentation_conflicts.length === 0 ? (
            <p className="answer-text muted">No documentation gaps or conflicts flagged in this window.</p>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {result.documentation_conflicts.map((c, i) => <li key={`c${i}`}><b>Possible conflict:</b> {c}</li>)}
              {result.documentation_gaps.map((g, i) => <li key={i}>{g}</li>)}
            </ul>
          )}
        </Section>
        <Section n={6} title="Relevant guideline context"><CitedText text={result.guideline_context} active={active} onCite={onCite} /></Section>
        <Section n={7} title="Patient-record evidence">
          <div className="controls">{result.patient_evidence.map((e) => (
            <button key={e.evidence_id} className="cite" aria-pressed={active === e.evidence_id} onClick={() => onCite(e.evidence_id)}>{e.evidence_id}</button>
          ))}{result.patient_evidence.length === 0 && <span className="muted">None cited.</span>}</div>
        </Section>
        <Section n={8} title="Guideline citations">
          <div className="controls">{result.guideline_citations.map((c) => (
            <button key={c.citation_id} className="cite guideline" aria-pressed={active === c.citation_id} onClick={() => onCite(c.citation_id)}>
              {c.citation_id} · {c.pages}
            </button>
          ))}{result.guideline_citations.length === 0 && <span className="muted">None cited.</span>}</div>
        </Section>
        <Section n={9} title="Limitations">
          <ul style={{ margin: 0, paddingLeft: 18 }}>{result.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
        </Section>
        {result.raw_model_output && (
          <details>
            <summary>Raw model output (before validation)</summary>
            <div className="table-wrap" style={{ marginTop: 8 }}>
              <table className="data">
                <tbody>
                  {Object.entries(result.raw_model_output).map(([k, v]) => (
                    <tr key={k}><td className="mono" style={{ whiteSpace: "nowrap" }}>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
        <p className="disclaimer">
          {result.disclaimer} · Retrieval: {result.retrieval_mode} · Evidence {fmtMs(result.timings_ms.evidence_and_retrieval)} · Generation {fmtMs(result.timings_ms.generation)} · Tokens {result.llm_tokens.prompt ?? "—"} in / {result.llm_tokens.generated ?? "—"} out
        </p>
      </div>
    </Card>
  );
}

function GuidelineAnswer({ result, active, onCite }) {
  return (
    <Card title="Guideline answer" sub={result.insufficient_evidence ? "Insufficient evidence in the indexed guidelines" : `${result.citations.length} verified citation(s)`}>
      <div className="stack" style={{ gap: 12 }}>
        <ValidationNotes result={result} />
        <CitedText text={result.answer} active={active} onCite={onCite} />
        <p className="disclaimer">{result.disclaimer} · Retrieval {fmtMs(result.timings_ms.retrieval)} · Generation {fmtMs(result.timings_ms.generation)}</p>
      </div>
    </Card>
  );
}

export default function Assistant({ patientId, hours, onHours, initial = {} }) {
  const [scope, setScope] = useState(initial.scope === "guidelines" ? "guidelines" : "patient");
  const [question, setQuestion] = useState(initial.q || EXAMPLES[0]);
  const [mode, setMode] = useState(MODES.some((m) => m.value === initial.mode) ? initial.mode : "hybrid");
  const autoRan = useRef(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [active, setActive] = useState(null);
  const refs = useRef({});
  const tl = useApi(() => (patientId ? api.timeline(patientId, hours) : Promise.resolve(null)), [patientId, hours]);

  useEffect(() => { setResult(null); setActive(null); }, [patientId, scope]);
  // Shareable links: #/assistant/P015?q=...&run=1 runs the question once on load.
  useEffect(() => {
    if (initial.run === "1" && !autoRan.current && (scope === "guidelines" || patientId)) {
      autoRan.current = true;
      submit();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);
  useEffect(() => {
    if (!busy) return undefined;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed((Date.now() - t0) / 1000), 200);
    return () => clearInterval(id);
  }, [busy]);

  const onCite = (id) => {
    setActive(id);
    refs.current[id]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  async function submit(e) {
    e?.preventDefault();
    if (!question.trim() || (scope === "patient" && !patientId)) return;
    setBusy(true); setError(null); setResult(null); setActive(null); setElapsed(0);
    try {
      const res = scope === "patient"
        ? await api.ask({ patient_id: patientId, question, hours: hours ?? undefined, retrieval_mode: mode })
        : await api.askGuidelines({ question, method: mode === "none" ? "hybrid" : mode });
      setResult(res);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      {scope === "patient" && tl.data && <PatientHeader timeline={tl.data} hours={hours} onHours={onHours} />}
      {scope === "patient" && !patientId && <Card><p className="empty">Select a synthetic patient from the list.</p></Card>}
      <Card title="DetoxRAG clinical assistant"
        sub="Answers are generated locally by Qwen3-4B from the retrieved records and guideline passages, then validated. Review support only — not clinical advice.">
        <form onSubmit={submit} className="stack" style={{ gap: 10 }}>
          <div className="controls">
            <div className="seg" role="group" aria-label="Question scope">
              <button type="button" aria-pressed={scope === "patient"} onClick={() => setScope("patient")}>About this patient</button>
              <button type="button" aria-pressed={scope === "guidelines"} onClick={() => setScope("guidelines")}>Guidelines only</button>
            </div>
            <select className="select" value={mode} onChange={(e) => setMode(e.target.value)} aria-label="Retrieval mode">
              {MODES.filter((m) => scope === "patient" || m.value !== "none").map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
          <textarea className="question" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={1000}
            aria-label="Question" onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(); }} />
          <div className="controls">
            {EXAMPLES.map((q) => <button type="button" key={q} className="chip" onClick={() => setQuestion(q)}>{q}</button>)}
          </div>
          <div className="controls">
            <button className="btn primary" type="submit" disabled={busy || !question.trim() || (scope === "patient" && !patientId)}>
              {busy ? "Generating…" : "Ask"}
            </button>
            {busy && <span className="muted"><span className="spinner" /> {elapsed.toFixed(1)} s — local model running (first request also loads models)</span>}
            <span className="muted" style={{ fontSize: 12 }}>⌘/Ctrl + Enter</span>
          </div>
        </form>
      </Card>
      <ErrorBox error={error} />
      {busy && <Card><Loading label="Retrieving records and guideline passages, then generating…" /></Card>}
      {result && (
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))", alignItems: "start" }}>
          {scope === "patient"
            ? <PatientAnswer result={result} active={active} onCite={onCite} />
            : <GuidelineAnswer result={result} active={active} onCite={onCite} />}
          <EvidencePanel result={result} active={active} refs={refs} />
        </div>
      )}
    </div>
  );
}
