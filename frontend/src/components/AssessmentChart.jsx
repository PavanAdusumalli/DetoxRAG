import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtAxisTime, fmtHours, fmtTime, niceTicks, toMs } from "../lib/format.js";
import { Card } from "./common.jsx";

export function timeTicks(start, end, target = 6) {
  const span = end - start;
  if (span <= 0) return [start];
  const hour = 3600e3;
  const steps = [1, 2, 3, 4, 6, 8, 12, 24].map((h) => h * hour);
  const step = steps.find((s) => span / s <= target) || 48 * hour;
  const first = Math.ceil(start / step) * step;
  const out = [];
  for (let t = first; t <= end; t += step) out.push(t);
  return out;
}

function ScoreTooltip({ active, payload, type }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tooltip">
      <div className="t">{fmtTime(p.x)}</div>
      <div className="row">
        <span>{type} score</span>
        <b>{p.score}</b>
      </div>
      <div className="row muted">
        <span>Record</span>
        <span className="mono">{p.id}</span>
      </div>
    </div>
  );
}

export default function AssessmentChart({ timeline }) {
  const [showTable, setShowTable] = useState(false);
  const { assessments, window, score_summary: s, intervals, documentation_gaps, documentation_conflicts } = timeline;
  const type = s.assessment_type;

  const data = useMemo(
    () => assessments.map((a) => ({ x: toMs(a.AssessmentDateTime), score: a.Score, id: a.AssessmentID })),
    [assessments],
  );
  const x0 = toMs(window.start);
  const x1 = toMs(window.end);
  const bands = [...intervals.long_intervals, intervals.leading_gap, intervals.trailing_gap].filter(Boolean);
  const vitalsOnly = documentation_gaps.filter((g) => g.gap_type === "vital_signs_without_assessment");
  const maxScore = Math.max(10, ...data.map((d) => d.score));
  const yTicks = niceTicks(0, maxScore + 2, 4);
  const yMax = yTicks[yTicks.length - 1];

  return (
    <Card
      title={`${type} assessment history`}
      sub={`Total score per recorded assessment (instrument range 0–${s.scale_max}). Shaded bands mark intervals longer than the ${fmtHours(intervals.flag_threshold_hours)} review flag — a software setting, not a clinical standard.`}
      actions={
        <button className="btn small" onClick={() => setShowTable((v) => !v)} aria-pressed={showTable}>
          {showTable ? "Show chart" : "Show table"}
        </button>
      }
    >
      <div className="legend" aria-hidden={showTable}>
        <span><span className="band" />No assessment recorded</span>
        {vitalsOnly.length > 0 && <span><span className="tick" />Vital signs recorded, no assessment</span>}
        {documentation_conflicts.length > 0 && <span>⚠ Possible documentation conflict</span>}
      </div>
      {data.length === 0 && <p className="empty">No {type} assessments are recorded in the retrieved dataset for this window.</p>}
      {showTable ? (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr><th>Time</th><th>Record</th><th className="num">{type} score</th></tr>
            </thead>
            <tbody>
              {assessments.map((a) => (
                <tr key={a.AssessmentID}>
                  <td className="tabular">{fmtTime(a.AssessmentDateTime)}</td>
                  <td className="mono">{a.AssessmentID}</td>
                  <td className="num">{a.Score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        data.length > 0 && (
          <div className="chart-box" role="img" aria-label={`${type} scores from ${s.earliest?.score} to ${s.latest?.score} over ${fmtHours(s.observation_span_hours)}`}>
            <ResponsiveContainer>
              <LineChart data={data} margin={{ top: 18, right: 16, bottom: 4, left: -8 }}>
                <CartesianGrid vertical={false} />
                <XAxis
                  dataKey="x" type="number" scale="time" domain={[x0, x1]}
                  ticks={timeTicks(x0, x1)} tickFormatter={fmtAxisTime}
                  stroke="var(--axis)" tickLine={false} minTickGap={24}
                />
                <YAxis domain={[0, yMax]} ticks={yTicks} allowDecimals={false} stroke="var(--axis)" tickLine={false} axisLine={false} width={44} />
                {bands.map((b, i) => (
                  <ReferenceArea
                    key={`band-${i}`} x1={toMs(b.start)} x2={toMs(b.end)} fill="var(--gap-band)" fillOpacity={1}
                    stroke="var(--axis)" strokeDasharray="3 3"
                    label={{ value: fmtHours(b.hours), position: "insideTop", fill: "var(--ink-2)", fontSize: 11 }}
                  />
                ))}
                {vitalsOnly.map((g) => (
                  <ReferenceLine key={g.record_ids[0]} x={toMs(g.start)} stroke="var(--ink-muted)" strokeDasharray="4 3" strokeWidth={1.5} />
                ))}
                {documentation_conflicts.map((c, i) => (
                  <ReferenceLine
                    key={`c-${i}`} x={toMs(c.time)} stroke="var(--serious)" strokeDasharray="2 3" strokeWidth={1.5}
                    label={{ value: "⚠", position: "top", fill: "var(--serious-ink)", fontSize: 13 }}
                  />
                ))}
                <Tooltip content={<ScoreTooltip type={type} />} cursor={{ stroke: "var(--axis)", strokeWidth: 1 }} />
                <Line
                  type="linear" dataKey="score" stroke="var(--series-1)" strokeWidth={2} isAnimationActive={false}
                  dot={{ r: 4, fill: "var(--series-1)", stroke: "var(--surface)", strokeWidth: 2 }}
                  activeDot={{ r: 6, stroke: "var(--surface)", strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )
      )}
    </Card>
  );
}
