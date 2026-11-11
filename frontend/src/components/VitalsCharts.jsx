import { useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtAxisTime, fmtTime, niceTicks, toMs } from "../lib/format.js";
import { timeTicks } from "./AssessmentChart.jsx";
import { Card } from "./common.jsx";

// One measure per chart (small multiples): never two y-scales on one chart.
const PANELS = [
  { key: "bp", title: "Blood pressure", unit: "mmHg", series: [
    { field: "SBP", label: "Systolic", color: "var(--series-1)" },
    { field: "DBP", label: "Diastolic", color: "var(--series-2)" },
  ] },
  { key: "pulse", title: "Pulse", unit: "beats/min", series: [{ field: "Pulse", label: "Pulse", color: "var(--series-1)" }] },
  { key: "temp", title: "Temperature", unit: "°F", series: [{ field: "Temperature", label: "Temperature", color: "var(--series-1)" }] },
  { key: "rr", title: "Respiratory rate", unit: "breaths/min", series: [{ field: "RespiratoryRate", label: "Resp. rate", color: "var(--series-1)" }] },
  { key: "spo2", title: "Oxygen saturation", unit: "%", series: [{ field: "OxygenSaturation", label: "SpO₂", color: "var(--series-1)" }] },
];

function VitalTooltip({ active, payload, panel }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="tooltip">
      <div className="t">{fmtTime(row.x)} · <span className="mono">{row.id}</span></div>
      {panel.series.map((s) => (
        <div className="row" key={s.field}>
          <span>{s.label}</span>
          <b>{row[s.field] ?? <span className="missing">not recorded</span>} {row[s.field] != null && panel.unit}</b>
        </div>
      ))}
    </div>
  );
}

function ticksFor(values, pad) {
  const v = values.filter((x) => x !== null && x !== undefined);
  if (!v.length) return [0, 1];
  return niceTicks(Math.min(...v) - pad, Math.max(...v) + pad, 3);
}

function Panel({ panel, data, x0, x1, summary }) {
  const values = panel.series.flatMap((s) => data.map((d) => d[s.field]));
  const pad = panel.key === "temp" ? 0.3 : panel.key === "spo2" ? 1 : 3;
  const yTicks = ticksFor(values, pad);
  const missing = panel.series.reduce((n, s) => n + data.filter((d) => d[s.field] == null).length, 0);
  const latest = panel.series.map((s) => summary[s.field]?.latest?.value).filter((v) => v != null);
  return (
    <div>
      <h3>
        {panel.title} <span className="muted" style={{ fontWeight: 400 }}>({panel.unit})</span>
        {latest.length > 0 && <span style={{ float: "right", color: "var(--ink)" }} className="tabular">latest {latest.join("/")}</span>}
      </h3>
      {panel.series.length > 1 && (
        <div className="legend">
          {panel.series.map((s) => (
            <span key={s.field}><span className="sw" style={{ background: s.color }} />{s.label}</span>
          ))}
        </div>
      )}
      <div className="chart-box small">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="x" type="number" scale="time" domain={[x0, x1]} ticks={timeTicks(x0, x1, 4)}
              tickFormatter={fmtAxisTime} stroke="var(--axis)" tickLine={false} minTickGap={30} />
            <YAxis domain={[yTicks[0], yTicks[yTicks.length - 1]]} ticks={yTicks} stroke="var(--axis)" tickLine={false}
              axisLine={false} width={48} tickFormatter={(v) => (Number.isInteger(v) ? v : v.toFixed(1))} />
            <Tooltip content={<VitalTooltip panel={panel} />} cursor={{ stroke: "var(--axis)" }} />
            {panel.series.map((s) => (
              <Line key={s.field} dataKey={s.field} stroke={s.color} strokeWidth={2} connectNulls={false}
                isAnimationActive={false} dot={{ r: 3, fill: s.color, stroke: "var(--surface)", strokeWidth: 1.5 }}
                activeDot={{ r: 5 }} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {missing > 0 && <p className="sub" style={{ margin: "2px 0 0" }}>{missing} value(s) not recorded — line is broken, not interpolated.</p>}
    </div>
  );
}

export default function VitalsCharts({ timeline }) {
  const [showTable, setShowTable] = useState(false);
  const { vitals, window, vitals_summary } = timeline;
  const data = useMemo(() => vitals.map((v) => ({ ...v, x: toMs(v.RecordedDateTime), id: v.VitalID })), [vitals]);
  const summary = Object.fromEntries(vitals_summary.measures.map((m) => [m.measure, m]));
  const x0 = toMs(window.start);
  const x1 = toMs(window.end);
  const cell = (v, unit) => (v == null ? <span className="missing">not recorded</span> : `${v}${unit ? ` ${unit}` : ""}`);

  return (
    <Card
      title="Vital signs"
      sub={`${vitals_summary.n_sets} vital-sign set(s) recorded in this window.`}
      actions={<button className="btn small" onClick={() => setShowTable((v) => !v)}>{showTable ? "Show charts" : "Show table"}</button>}
    >
      {vitals.length === 0 ? (
        <p className="empty">No vital signs are recorded in the retrieved dataset for this window.</p>
      ) : showTable ? (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Time</th><th>Record</th><th className="num">BP (mmHg)</th><th className="num">Pulse (/min)</th>
                <th className="num">Temp (°F)</th><th className="num">RR (/min)</th><th className="num">SpO₂ (%)</th>
              </tr>
            </thead>
            <tbody>
              {vitals.map((v) => (
                <tr key={v.VitalID}>
                  <td className="tabular">{fmtTime(v.RecordedDateTime)}</td>
                  <td className="mono">{v.VitalID}</td>
                  <td className="num">{v.SBP == null || v.DBP == null ? cell(null) : `${v.SBP}/${v.DBP}`}</td>
                  <td className="num">{cell(v.Pulse)}</td>
                  <td className="num">{cell(v.Temperature)}</td>
                  <td className="num">{cell(v.RespiratoryRate)}</td>
                  <td className="num">{cell(v.OxygenSaturation)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="grid grid-3">
          {PANELS.map((p) => <Panel key={p.key} panel={p} data={data} x0={x0} x1={x1} summary={summary} />)}
        </div>
      )}
    </Card>
  );
}
