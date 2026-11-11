// Thin client for the DetoxRAG FastAPI backend.
const BASE = import.meta.env.VITE_API_BASE || "";

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError(0, "Cannot reach the DetoxRAG backend. Is `uvicorn backend.main:app` running on port 8000?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

function qs(params) {
  const p = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  return p.length ? "?" + new URLSearchParams(p).toString() : "";
}

export const api = {
  health: () => request("/api/health"),
  patients: (withdrawalType) => request("/api/patients" + qs({ withdrawal_type: withdrawalType })),
  patient: (id) => request(`/api/patients/${encodeURIComponent(id)}`),
  timeline: (id, hours) => request(`/api/patients/${encodeURIComponent(id)}/timeline` + qs({ hours })),
  ask: (body) => request("/api/rag/ask", { method: "POST", body: JSON.stringify(body) }),
  askGuidelines: (body) => request("/api/guidelines/ask", { method: "POST", body: JSON.stringify(body) }),
  guidelines: () => request("/api/guidelines"),
  passage: (id) => request(`/api/guidelines/passages/${encodeURIComponent(id)}`),
  evaluation: () => request("/api/evaluation/results"),
};
