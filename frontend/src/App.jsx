import { useState } from "react";
import PatientList from "./components/PatientList.jsx";
import { useApi, useHashRoute } from "./hooks/useApi.js";
import Assistant from "./pages/Assistant.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Evaluation from "./pages/Evaluation.jsx";
import { api } from "./services/api.js";

const PAGES = [
  { id: "dashboard", label: "Monitoring dashboard" },
  { id: "assistant", label: "Clinical assistant" },
  { id: "evaluation", label: "RAG evaluation" },
];

export default function App() {
  const [route, navigate] = useHashRoute();
  const [hours, setHours] = useState(() => (route.params.hours ? Number(route.params.hours) : null));
  const patients = useApi(() => api.patients(), []);
  const health = useApi(() => api.health(), []);
  const patientId = route.patientId || patients.data?.[0]?.PatientID || null;
  const page = PAGES.some((p) => p.id === route.page) ? route.page : "dashboard";
  const showSidebar = page !== "evaluation";

  return (
    <>
      <div className="banner" role="note">
        SYNTHETIC DATA · All patients and records are fictional · Research demonstration — not a clinical decision-support system
      </div>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 32 32"><path d="M3 17h6l3-8 6 16 3-8h8" stroke="#fff" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </div>
          <div>
            <h1>DetoxRAG</h1>
            <p>AI-assisted withdrawal monitoring review</p>
          </div>
        </div>
        <nav className="tabs" aria-label="Screens">
          {PAGES.map((p) => (
            <button key={p.id} className="tab" aria-current={page === p.id ? "page" : undefined}
              onClick={() => navigate(p.id, p.id === "evaluation" ? null : patientId)}>
              {p.label}
            </button>
          ))}
        </nav>
        <div className="health" title={health.data ? JSON.stringify(health.data.models) : ""}>
          <span className={`dot ${health.data?.status === "ok" ? "ok" : health.error ? "bad" : ""}`} />
          {health.data ? `Backend ${health.data.status} · local models ${health.data.models.llm_loaded ? "loaded" : "on demand"}` : health.error ? "Backend unreachable" : "Checking…"}
        </div>
      </header>
      <main className={`layout ${showSidebar ? "" : "wide"}`}>
        {showSidebar && (
          <PatientList patients={patients.data} loading={patients.loading} error={patients.error} reload={patients.reload}
            selected={patientId} onSelect={(id) => navigate(page, id)} />
        )}
        <div className="stack">
          {page === "dashboard" && (
            <Dashboard patientId={patientId} hours={hours} onHours={setHours} onAsk={() => navigate("assistant", patientId)} />
          )}
          {page === "assistant" && <Assistant patientId={patientId} hours={hours} onHours={setHours} initial={route.params} />}
          {page === "evaluation" && <Evaluation />}
        </div>
      </main>
    </>
  );
}
