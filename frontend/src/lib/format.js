// Pure formatting helpers (unit-tested in format.test.js).

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Backend timestamps are facility-local "YYYY-MM-DDTHH:MM:SS" with no zone. */
export function parseTs(ts) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/.exec(ts || "");
  if (!m) return null;
  return new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +(m[6] || 0));
}

export function toMs(ts) {
  const d = parseTs(ts);
  return d ? d.getTime() : null;
}

const pad = (n) => String(n).padStart(2, "0");

export function fmtTime(ts) {
  const d = typeof ts === "number" ? new Date(ts) : parseTs(ts);
  if (!d) return "—";
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function fmtClock(ms) {
  const d = new Date(ms);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function fmtAxisTime(ms) {
  const d = new Date(ms);
  return `${MONTHS[d.getMonth()]} ${d.getDate()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function fmtHours(h) {
  if (h === null || h === undefined) return "—";
  return `${Number(h).toFixed(h >= 10 ? 1 : 2).replace(/\.?0+$/, "")} h`;
}

export function fmtSigned(n) {
  if (n === null || n === undefined) return "—";
  return n > 0 ? `+${n}` : `${n}`;
}

export function fmtPct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

export function fmtMs(ms) {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1) return "<1 ms";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

/** Round axis ticks covering [min, max] with about `count` intervals (1/2/2.5/5 × 10^n steps). */
export function niceTicks(min, max, count = 4) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) { min -= 1; max += 1; }
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const out = [];
  for (let v = lo; v <= hi + step / 2; v += step) out.push(Number(v.toFixed(6)));
  return out;
}

/** Split text into plain segments and citation segments: "a [X-1, Y-2] b". */
export function splitCitations(text) {
  if (!text) return [];
  const out = [];
  const re = /\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+(?:\s*[,;]\s*[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)*)\]/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ type: "text", value: text.slice(last, m.index) });
    for (const id of m[1].split(/\s*[,;]\s*/)) out.push({ type: "cite", value: id });
    last = re.lastIndex;
  }
  if (last < text.length) out.push({ type: "text", value: text.slice(last) });
  return out;
}

/** Guideline chunk IDs look like BJA-2023-0071; everything else is patient evidence. */
export function isGuidelineId(id) {
  return /^[A-Z]{2,5}-\d{4}-\d{4}$/.test(id);
}

/** Medication status -> reserved status tone + icon + label (never color alone). */
export function statusMeta(status) {
  switch (status) {
    case "Administered":
      return { tone: "good", icon: "✓", label: "Administered" };
    case "Held":
      return { tone: "warning", icon: "‖", label: "Held" };
    case "Refused":
      return { tone: "serious", icon: "✕", label: "Refused" };
    case "Not administered":
      return { tone: "serious", icon: "–", label: "Not administered" };
    case "Not documented":
      return { tone: "critical", icon: "?", label: "Not documented" };
    default:
      return { tone: "neutral", icon: "•", label: status || "Unknown" };
  }
}

export const GAP_LABELS = {
  no_assessments_in_window: "No assessments in window",
  no_assessment_at_window_start: "No assessment at window start",
  long_assessment_interval: "Long assessment interval",
  no_assessment_at_window_end: "Records stop before window end",
  assessment_without_vital_signs: "Assessment without vital signs",
  vital_signs_without_assessment: "Vital signs without assessment",
  missing_vital_measurement: "Missing vital measurement",
  medication_not_documented: "Dose not documented",
};
