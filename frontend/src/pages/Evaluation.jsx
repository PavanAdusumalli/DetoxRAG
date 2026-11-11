import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, ErrorBox, Loading, Tile } from "../components/common.jsx";
import { useApi } from "../hooks/useApi.js";
import { fmtMs, fmtPct } from "../lib/format.js";
import { api } from "../services/api.js";

// Fixed slot per configuration (color follows the entity, never its rank).
const SLOT = {
  dense: "var(--series-1)", bm25: "var(--series-2)", hybrid: "var(--series-3)",
  A_structured_only: "var(--series-2)", B_dense_rag: "var(--series-1)", C_hybrid_rag: "var(--series-3)",
};

const RETRIEVAL_METRICS = [
  ["recall_at_k", "Recall@K"], ["precision_at_k", "Precision@K"], ["mrr", "MRR"], ["hit_at_1", "Hit@1"],
];
const GENERATION_METRICS = [
  ["citation_validity", "Citation validity", "Share of model-written citation IDs that matched supplied evidence"],
  ["patient_fact_accuracy", "Patient fact accuracy", "Gold facts (earliest, latest, change) stated correctly, of those stated"],
  ["patient_fact_coverage", "Patient fact coverage", "Gold facts stated at all"],
  ["contradiction_free_rate", "No contradicted values", "Cases where no numeric claim was contradicted by the computed analytics"],
  ["unsupported_sentence_rate", "Unsupported sentence rate", "Patient sentences with no valid citation or a contradicted value (lower is better)"],
  ["gap_acknowledgement_rate", "Missing-evidence handling", "Cases with documented gaps where the text acknowledges them"],
  ["guideline_evidence_success", "Evidence retrieval success", "Cases where ≥1 gold-relevant guideline passage was retrieved"],
  ["guideline_citation_relevance", "Guideline citation relevance", "Cited passages that are gold-relevant"],
  ["safety_removal_rate", "Safety removals", "Cases where a severity/recommendation sentence was removed (lower is better)"],
  ["json_valid_rate", "Valid JSON first try", ""],
];
const QA_METRICS = [
  ["answered_rate", "Answered"], ["citation_validity", "Citation validity"],
  ["cited_gold_rate", "Cites a gold passage"], ["gold_retrieved_rate", "Gold passage retrieved"],
];

function MetricBars({ rows, keys, height = 240, percent = true }) {
  const data = keys.map(([k, label]) => ({ metric: label, ...Object.fromEntries(rows.map((r) => [r.key, r.metrics[k]])) }));
  return (
    <>
      <div className="legend">
        {rows.map((r) => <span key={r.key}><span className="sw" style={{ background: SLOT[r.key] }} />{r.label}</span>)}
      </div>
      <div className="chart-box" style={{ height }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }} barGap={2} barCategoryGap="22%">
            <CartesianGrid vertical={false} />
            <XAxis dataKey="metric" stroke="var(--axis)" tickLine={false} interval={0} />
            <YAxis domain={percent ? [0, 1] : [0, "auto"]} tickFormatter={(v) => (percent ? `${Math.round(v * 100)}%` : v)}
              stroke="var(--axis)" tickLine={false} axisLine={false} width={48} />
            <Tooltip cursor={{ fill: "var(--gap-band)" }}
              content={({ active, payload, label }) => active && payload?.length ? (
                <div className="tooltip">
                  <div className="t">{label}</div>
                  {payload.map((p) => (
                    <div className="row" key={p.dataKey}>
                      <span>{rows.find((r) => r.key === p.dataKey)?.label}</span>
                      <b>{p.value == null ? "n/a" : percent ? fmtPct(p.value) : p.value.toFixed(3)}</b>
                    </div>
                  ))}
                </div>
              ) : null} />
            {rows.map((r) => <Bar key={r.key} dataKey={r.key} fill={SLOT[r.key]} radius={[4, 4, 0, 0]} isAnimationActive={false} />)}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}

function MetricTable({ rows, keys, fmt = fmtPct, extra = [] }) {
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr><th>Configuration</th>{keys.map(([k, label, help]) => <th key={k} className="num" title={help}>{label}</th>)}{extra.map(([k, l]) => <th key={k} className="num">{l}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <td><span className="sw" style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2, background: SLOT[r.key], marginRight: 8 }} />{r.label}</td>
              {keys.map(([k]) => <td key={k} className="num">{r.metrics[k] == null ? "n/a" : fmt(r.metrics[k])}</td>)}
              {extra.map(([k, , f]) => <td key={k} className="num">{f(r.metrics[k])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const toRows = (obj) => Object.entries(obj || {}).map(([key, v]) => ({ key, label: v.label || key, metrics: v.metrics || v }));

export default function Evaluation() {
  const ev = useApi(() => api.evaluation(), []);
  const [showCases, setShowCases] = useState(false);
  if (ev.loading) return <Card><Loading label="Loading evaluation results…" /></Card>;
  if (ev.error?.status === 404)
    return (
      <Card title="RAG evaluation">
        <p className="empty">No benchmark results yet. Run <code>python -m evaluation.run_evaluation</code> to produce them — nothing here is hardcoded.</p>
      </Card>
    );
  if (ev.error) return <ErrorBox error={ev.error} onRetry={ev.reload} />;
  const r = ev.data;
  const retrieval = toRows(r.retrieval?.methods);
  const generation = toRows(r.generation?.experiments);
  const qa = toRows(r.guideline_qa?.experiments);
  const gd = r.analytics?.gap_detection;

  return (
    <div className="stack">
      <Card title="RAG evaluation" sub={`Run ${r.run_id} · ${r.created_at} · ${r.environment?.machine ?? ""} · ${Math.round(r.duration_s / 60)} min`}>
        <div className="tiles">
          <Tile label="Retrieval questions" value={r.retrieval?.n_questions ?? "—"} note={`K = ${r.retrieval?.k}`} />
          <Tile label="Patient cases" value={r.generation?.n_cases ?? "—"} note={`${generation.length} configurations`} />
          <Tile label="Guideline Q&A questions" value={r.guideline_qa?.n_questions ?? "—"} />
          <Tile label="Injected defects detected" value={gd ? `${gd.detected}/${gd.injected_defects}` : "—"} note={gd ? `${gd.flags_on_clean_bookings} flags on clean bookings` : null} />
          <Tile label="Temporal calc. accuracy" value={fmtPct(r.analytics?.temporal_accuracy?.accuracy)} note={r.analytics?.temporal_accuracy ? `${r.analytics.temporal_accuracy.n_checks} checks` : null} />
          <Tile label="Patient-data isolation" value={r.isolation ? (r.isolation.violations === 0 ? "0 leaks" : `${r.isolation.violations} leaks`) : "—"} note={r.isolation ? `${r.isolation.n_checks} checks` : null} />
        </div>
        {r.caveats?.length > 0 && (
          <div className="callout warn" style={{ marginTop: 12 }}>
            <strong>Read with care</strong>
            <ul>{r.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
          </div>
        )}
      </Card>

      {retrieval.length > 0 && (
        <Card title="Retrieval: dense vs BM25 vs hybrid" sub={`${r.retrieval.n_questions} guideline questions with manually reviewed gold passages (by recommendation ID / section). K = ${r.retrieval.k}.`}>
          <MetricBars rows={retrieval} keys={RETRIEVAL_METRICS} />
          <MetricTable rows={retrieval} keys={RETRIEVAL_METRICS} fmt={(v) => v.toFixed(3)} extra={[["mean_latency_ms", "Latency", fmtMs]]} />
        </Card>
      )}

      {generation.length > 0 && (
        <Card title="Generation: Experiments A–C" sub="A = structured data only · B = + dense RAG · C = + hybrid RAG. Metrics on the raw model output, before the validator's corrections.">
          <MetricBars rows={generation} keys={GENERATION_METRICS.slice(0, 7).map(([k, l]) => [k, l.replace("Missing-evidence handling", "Gap handling")])} height={280} />
          <MetricTable rows={generation} keys={GENERATION_METRICS}
            extra={[["mean_generation_ms", "Gen. latency", fmtMs], ["mean_total_ms", "Total latency", fmtMs]]} />
        </Card>
      )}

      {qa.length > 0 && (
        <Card title="Guideline-only Q&A" sub="Cited answers to the retrieval questions, dense vs hybrid.">
          <MetricTable rows={qa} keys={QA_METRICS} extra={[["mean_total_ms", "Latency", fmtMs]]} />
        </Card>
      )}

      <div className="grid grid-2">
        {gd && (
          <Card title="Deterministic gap detection" sub="Against the generator's injected defects (synthetic_ground_truth).">
            <div className="table-wrap">
              <table className="data">
                <thead><tr><th>Defect type</th><th className="num">Injected</th><th className="num">Detected</th><th className="num">Recall</th></tr></thead>
                <tbody>
                  {Object.entries(gd.by_type).map(([k, v]) => (
                    <tr key={k}><td>{k.replaceAll("_", " ")}</td><td className="num">{v.expected}</td><td className="num">{v.detected}</td><td className="num">{fmtPct(v.recall)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
        {r.system && (
          <Card title="System" sub="Measured on the development machine during this run.">
            <div className="table-wrap">
              <table className="data">
                <tbody>
                  {Object.entries(r.system).map(([k, v]) => (
                    <tr key={k}><td>{k.replaceAll("_", " ")}</td><td className="num">{typeof v === "number" ? (k.endsWith("_ms") ? fmtMs(v) : v.toLocaleString()) : String(v)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        )}
      </div>

      {r.not_run?.length > 0 && (
        <Card title="Not run">
          <ul style={{ margin: 0, paddingLeft: 18 }}>{r.not_run.map((n, i) => <li key={i}><b>{n.experiment}:</b> {n.reason}</li>)}</ul>
        </Card>
      )}

      {r.manual_review?.length > 0 && (
        <Card title="Manual review notes" sub="Human inspection of a sample of generated summaries.">
          <ul style={{ margin: 0, paddingLeft: 18 }}>{r.manual_review.map((m, i) => <li key={i}><b>{m.case}</b>: {m.notes}</li>)}</ul>
        </Card>
      )}

      {r.generation?.per_case?.length > 0 && (
        <Card title="Per-case results" actions={<button className="btn small" onClick={() => setShowCases((v) => !v)}>{showCases ? "Hide" : "Show"}</button>}>
          {showCases && (
            <div className="table-wrap">
              <table className="data">
                <thead><tr><th>Case</th><th>Exp.</th><th>Patient</th><th className="num">Facts ok</th><th className="num">Invalid cites</th><th className="num">Fact-check fails</th><th className="num">Uncited</th><th className="num">Latency</th></tr></thead>
                <tbody>
                  {r.generation.per_case.map((c, i) => (
                    <tr key={i}>
                      <td>{c.case_id}</td><td>{c.experiment.split("_")[0]}</td><td className="mono">{c.patient_id}</td>
                      <td className="num">{c.facts_correct}/{c.facts_total}</td><td className="num">{c.invalid_citations}</td>
                      <td className="num">{c.fact_check_failures}</td><td className="num">{c.uncited_sentences}/{c.sentences}</td>
                      <td className="num">{fmtMs(c.total_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
