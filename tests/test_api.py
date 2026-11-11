"""HTTP API tests (FastAPI TestClient) with a temporary database and fake models."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api import deps
from backend.api.routes import system
from backend.database.db import connect_readonly
from backend.database.synthetic_generator import GeneratorConfig, generate_dataset, write_sqlite
from backend.main import create_app
from backend.rag.llm import GenerationResult
from tests.conftest import make_chunk, make_passage, requires_index, requires_llm

GOOD_JSON = json.dumps({
    "patient_summary": "Monitoring was recorded [CALC-SCORE].",
    "assessment_trend": "Scores are summarized [CALC-SCORE].",
    "vital_sign_observations": "Vital signs were recorded [CALC-VITALS].",
    "medication_documentation": "Doses are documented [CALC-MEDS].",
    "guideline_context": "Assess at least every 8 hours [BJA-2023-0071].",
})


class FakeLLM:
    def __init__(self, text=GOOD_JSON):
        self.text = text
        self.calls = 0

    def generate(self, messages, max_tokens=None, temperature=None):
        self.calls += 1
        return GenerationResult(self.text, 1000, 50, 0.1, 50.0, 2.0)


class FakeRetriever:
    def search(self, query, k):
        return [make_passage(make_chunk("BJA-2023-0071", "A-13. Assess with the CIWA-Ar at least every 8 hours."))][:k]


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("api") / "api.db"
    write_sqlite(generate_dataset(GeneratorConfig()), p)
    return p


@pytest.fixture()
def fake_llm():
    return FakeLLM()


@pytest.fixture()
def client(db_path, fake_llm):
    app = create_app()

    def db():
        c = connect_readonly(db_path)
        try:
            yield c
        finally:
            c.close()

    app.dependency_overrides[deps.get_db] = db
    app.dependency_overrides[deps.get_llm] = lambda: fake_llm
    app.dependency_overrides[deps.get_retriever_factory] = lambda: (lambda mode: None if mode == "none" else FakeRetriever())
    return TestClient(app)


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert {"database", "guideline_index", "models", "version"} <= set(body)
    assert r.headers["x-request-id"]


def test_openapi_docs_available(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/health", "/api/patients", "/api/patients/{patient_id}", "/api/patients/{patient_id}/timeline",
              "/api/rag/ask", "/api/guidelines", "/api/evaluation/results"):
        assert p in paths


def test_evaluation_results_missing_then_present(client, tmp_path, monkeypatch):
    monkeypatch.setattr(system, "RESULTS_DIR", tmp_path)
    assert client.get("/api/evaluation/results").status_code == 404
    (tmp_path / "latest.json").write_text(json.dumps({"run_id": "x", "retrieval": {}}))
    assert client.get("/api/evaluation/results").json()["run_id"] == "x"


# --------------------------------------------------------------------------- #
# Patients
# --------------------------------------------------------------------------- #


def test_list_patients(client):
    r = client.get("/api/patients")
    assert r.status_code == 200 and len(r.json()) == 100
    alcohol = client.get("/api/patients", params={"withdrawal_type": "Alcohol"}).json()
    assert len(alcohol) == 50 and {p["AssessmentType"] for p in alcohol} == {"CIWA-Ar"}
    assert client.get("/api/patients", params={"withdrawal_type": "Cocaine"}).status_code == 422


def test_patient_detail(client):
    r = client.get("/api/patients/P015")
    assert r.status_code == 200
    body = r.json()
    assert body["booking"]["BookingNumber"] == "B1015" and body["monitoring_type"] == "CIWA-Ar"
    assert body["record_counts"]["withdrawal_assessments"] == 12
    assert "SYNTHETIC" in body["synthetic_data_notice"]


def test_timeline_window_and_analytics(client):
    r = client.get("/api/patients/P015/timeline", params={"hours": 12})
    assert r.status_code == 200
    t = r.json()
    assert t["window"]["hours"] == 12.0
    assert all(t["window"]["start"] <= a["AssessmentDateTime"] <= t["window"]["end"] for a in t["assessments"])
    assert t["score_summary"]["assessment_type"] == "CIWA-Ar"
    assert {"documentation_gaps", "documentation_conflicts", "events", "vitals_summary", "medication_summary"} <= set(t)


@pytest.mark.parametrize("url,code", [
    ("/api/patients/P9/timeline", 422),                         # bad ID format
    ("/api/patients/P001%3B%20DROP/timeline", 422),             # injection attempt
    ("/api/patients/P999/timeline", 404),                       # unknown patient
    ("/api/patients/P001/timeline?booking_number=B1002", 404),  # another patient's booking
    ("/api/patients/P001/timeline?booking_number=XYZ", 422),
    ("/api/patients/P001/timeline?hours=-5", 422),
    ("/api/patients/P001/timeline?hours=10000", 422),
    ("/api/patients/P001/timeline?start=yesterday", 422),
    ("/api/patients/P001/timeline?start=2026-03-25T00:00:00&end=2026-03-24T00:00:00", 422),
])
def test_invalid_requests(client, url, code):
    r = client.get(url)
    assert r.status_code == code, r.text
    assert "detail" in r.json()


def test_no_cross_patient_leakage_in_timeline(client, db_path):
    conn = connect_readonly(db_path)
    for pid, bk in conn.execute("SELECT PatientID, BookingNumber FROM patient_bookings LIMIT 25"):
        t = client.get(f"/api/patients/{pid}/timeline").json()
        assert t["booking"]["PatientID"] == pid and t["booking"]["BookingNumber"] == bk
        own = {r[0] for r in conn.execute("SELECT AssessmentID FROM withdrawal_assessments WHERE BookingNumber = ?", (bk,))}
        assert {a["AssessmentID"] for a in t["assessments"]} == own
        own_notes = {r[0] for r in conn.execute("SELECT NoteID FROM nursing_notes WHERE BookingNumber = ?", (bk,))}
        assert {n["NoteID"] for n in t["notes"]} == own_notes
    conn.close()


# --------------------------------------------------------------------------- #
# Assistant
# --------------------------------------------------------------------------- #


def test_rag_ask(client, fake_llm):
    r = client.post("/api/rag/ask", json={"patient_id": "P015", "question": "Summarize the last 12 hours.", "hours": 12})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["patient_id"], body["booking_number"], body["monitoring_type"]) == ("P015", "B1015", "CIWA-Ar")
    for field in ("patient_summary", "assessment_trend", "vital_sign_observations", "medication_documentation",
                  "documentation_gaps", "guideline_context", "patient_evidence", "guideline_citations", "limitations"):
        assert field in body
    assert [c["citation_id"] for c in body["guideline_citations"]] == ["BJA-2023-0071"]
    assert {e["evidence_id"] for e in body["patient_evidence"]} == {"CALC-SCORE", "CALC-VITALS", "CALC-MEDS"}
    assert body["disclaimer"] and fake_llm.calls == 1


def test_rag_ask_without_retrieval(client):
    body = client.post("/api/rag/ask", json={"patient_id": "P002", "question": "What is missing?",
                                             "retrieval_mode": "none"}).json()
    assert body["retrieval_mode"] == "none" and body["retrieved_passages"] == []
    assert body["documentation_gaps"]  # deterministic gaps always present


@pytest.mark.parametrize("payload,code", [
    ({"patient_id": "P015", "question": ""}, 422),
    ({"patient_id": "P015", "question": "x" * 1001}, 422),
    ({"patient_id": "15", "question": "Summarize"}, 422),
    ({"patient_id": "P015", "question": "Summarize", "retrieval_mode": "web"}, 422),
    ({"patient_id": "P015", "booking_number": "B1002", "question": "Summarize"}, 404),
    ({"patient_id": "P999", "question": "Summarize"}, 404),
    ({"patient_id": "P015", "question": "   "}, 400),
])
def test_rag_ask_invalid(client, payload, code):
    assert client.post("/api/rag/ask", json=payload).status_code == code


def test_rag_ask_model_unavailable_returns_503(db_path):
    app = create_app()

    def db():
        c = connect_readonly(db_path)
        yield c
        c.close()

    def missing_llm():
        raise FileNotFoundError("LLM not found at models/qwen3-4b-mlx-4bit")

    app.dependency_overrides[deps.get_db] = db
    app.dependency_overrides[deps.get_llm] = missing_llm
    r = TestClient(app).post("/api/rag/ask", json={"patient_id": "P015", "question": "Summarize"})
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]


def test_guidelines_ask_with_fakes(client, fake_llm):
    fake_llm.text = json.dumps({"answer": "Every 8 h [BJA-2023-0071].", "citations": ["BJA-2023-0071"],
                                "insufficient_evidence": False})
    body = client.post("/api/guidelines/ask", json={"question": "How often CIWA-Ar?"}).json()
    assert body["citations"][0]["citation_id"] == "BJA-2023-0071"


# --------------------------------------------------------------------------- #
# Real index / model
# --------------------------------------------------------------------------- #


@requires_index
@pytest.mark.integration
def test_guideline_listing_and_passage():
    c = TestClient(create_app())
    g = c.get("/api/guidelines").json()
    assert {d["document_id"] for d in g["documents"]} == {"BJA-2023", "BOP-2020"}
    assert g["not_included"] and g["index"]["n_chunks"] > 100
    p = c.get("/api/guidelines/passages/BJA-2023-0071")
    assert p.status_code == 200 and "A-13" in p.json()["recommendation_ids"]
    assert c.get("/api/guidelines/passages/BJA-2023-9999").status_code == 404
    assert c.get("/api/guidelines/passages/bad").status_code == 422


@requires_index
@pytest.mark.integration
def test_guideline_search_real():
    hits = TestClient(create_app()).get("/api/guidelines/search", params={"q": "COWS fentanyl", "method": "bm25"}).json()
    assert hits and hits[0]["method"] == "bm25"


@requires_index
@requires_llm
@pytest.mark.llm
@pytest.mark.integration
def test_rag_ask_real_models(db_path):
    app = create_app()

    def db():
        c = connect_readonly(db_path)
        yield c
        c.close()

    app.dependency_overrides[deps.get_db] = db
    r = TestClient(app).post("/api/rag/ask", json={"patient_id": "P015", "hours": 12,
                                                   "question": "Summarize the withdrawal progression and relevant guidelines."})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["generated"] and body["validation"]["invalid_citations"] == []
    assert body["retrieved_passages"]
