"""
MediTrace - FastAPI Clinical Workflow Integration Test Suite.

Verifies Phase 5 integration:
1. Health endpoint functionality.
2. Successful consultation invokes the injected LangGraph workflow.
3. Real engine adapter (LangGraphClinicalEngine) delegates to LangGraph.
4. SOAP draft fields (subjective, objective, assessment, plan) are preserved.
5. Safety review flags reach the API response in both safety_flags and safety_alerts.
6. Missing information records reach the API response.
7. Reference provenance (source_document, relevance_score, passage_id) reaches response.
8. Draft lifecycle status and mandatory clinical safety disclaimer are preserved.
9. Empty and whitespace consultation text are properly rejected (422 and 400).
10. Clinical extraction failure halts workflow without producing a fabricated note (502).
11. Retrieval with no evidence is distinguished from retrieval failure.
12. Safety verification failure halts workflow and does not silently continue (500).
13. SOAP generation failure produces an explicit error without fabricating notes (502).
14. Missing live configuration (GEMINI_API_KEY) in live mode produces actionable error (503).
15. Mock engine is used only when explicitly selected.
16. Live mode never silently falls back to mock output on error.
17. Transcription endpoint remains fully functional.
18. Mock transcription remains test-only and prominently identified.
19. Zero live Gemini API network calls occur during testing.
20. No GEMINI_API_KEY is required for offline tests with injected services.
"""

from io import BytesIO
import os
import socket
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.schemas import (
    CLINICAL_SAFETY_DISCLAIMER,
    AllergyInfo,
    ClinicalExtractionResult,
    CertaintyLevel,
    DraftSOAPNote,
    InformationSource,
    MedicationMention,
    MedicationStatus,
    MissingInformation,
    NoteStatus,
    PresenceStatus,
    ReferencePassage,
    ReviewStatus,
    SafetyCategory,
    SafetyReviewFlag,
    SafetySeverity,
    SymptomInfo,
)
from app.services.clinical_extraction import (
    ClinicalExtractionService,
    ExtractionServiceError,
)
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import GeminiLLMService
from app.services.retrieval_service import RetrievalService, RetrievalServiceError
from app.services.safety_verification import (
    ClinicalSafetyVerificationService,
    SafetyVerificationError,
)
from app.services.soap_note_service import (
    SOAPNoteGenerationError,
    SOAPNoteService,
)
from app.services.vector_store import VectorStore
from backend.config import settings
from backend.main import app
from backend.services.clinical_ai import (
    ClinicalEngineConfigurationError,
    ClinicalWorkflowExecutionError,
    LangGraphClinicalEngine,
    MockClinicalEngine,
    get_clinical_engine,
)
from backend.services.transcription import (
    MockTranscriptionEngine,
    get_transcription_engine,
)
from backend.workflows.clinical_graph import (
    ClinicalWorkflow,
    ClinicalWorkflowServices,
)

client = TestClient(app)

SYNTHETIC_TRANSCRIPT = (
    "Doctor: Good morning Jane, what brings you in today?\n"
    "Patient: I've had a severe sore throat, body aches, and fever for the past 2 days. It hurts to swallow.\n"
    "Doctor: Let me check your throat. There is marked erythema and tonsillar exudate. "
    "I also see documented in your chart that you had a severe rash and hives from Penicillin in the past. "
    "I am considering prescribing Amoxicillin 500mg three times daily.\n"
    "Patient: Yes, that sounds good if it will help."
)

DUMMY_WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


@pytest.fixture
def synthetic_extraction():
    return ClinicalExtractionResult(
        encounter_id="ENC-SYNTHETIC-FASTAPI",
        symptoms=[
            SymptomInfo(
                name="Sore throat",
                presence=PresenceStatus.PRESENT,
                certainty=CertaintyLevel.CONFIRMED,
                source=InformationSource.PATIENT_REPORTED,
                duration="2 days",
                severity="severe",
            ),
            SymptomInfo(
                name="Fever",
                presence=PresenceStatus.PRESENT,
                certainty=CertaintyLevel.SUSPECTED,
                source=InformationSource.PATIENT_REPORTED,
                duration="2 days",
            ),
        ],
        allergies=[
            AllergyInfo(
                allergen="Penicillin",
                reaction_type="severe rash and hives",
                source=InformationSource.HISTORICAL_EHR,
                certainty=CertaintyLevel.CONFIRMED,
            )
        ],
        medications=[
            MedicationMention(
                name="Amoxicillin",
                dosage="500mg",
                frequency="three times daily",
                status=MedicationStatus.PROPOSED,
                route="oral",
            )
        ],
        missing_information=[
            MissingInformation(
                category="vital_signs",
                description="Objective measured body temperature not documented",
                clinical_relevance="Pre-prescription allergy screening and objective fever measurement are essential.",
            )
        ],
    )


@pytest.fixture
def synthetic_draft_soap(synthetic_extraction):
    return DraftSOAPNote(
        note_id="SOAP-SYNTHETIC-FASTAPI-01",
        subjective=(
            "Patient is a female reporting severe sore throat, feverishness, and body aches for 2 days. "
            "Reports pain with swallowing. Confirmed historical allergy to Penicillin (rash, hives)."
        ),
        objective=(
            "Pharynx exam reveals marked pharyngeal erythema and bilateral tonsillar exudate. "
            "No measured vital signs documented."
        ),
        assessment=(
            "Acute tonsillopharyngitis, clinically suspect for Group A Streptococcus. "
            "Proposed Amoxicillin is CONTRAINDICATED due to documented Penicillin allergy."
        ),
        plan=(
            "1. Hold Amoxicillin prescription due to cross-reactivity risk. "
            "2. Select macrolide or clindamycin alternative after clinician review. "
            "3. Obtain rapid strep throat swab."
        ),
        status=NoteStatus.DRAFT_PENDING_REVIEW,
        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
        missing_information=synthetic_extraction.missing_information,
    )


@pytest.fixture
def in_memory_retrieval_service():
    retrieval_svc = MagicMock(spec=RetrievalService)
    passages = [
        ReferencePassage(
            passage_id="ref-fastapi-01",
            title="Allergy Alternatives",
            content="For patients with documented penicillin allergy, beta-lactam antibiotics are contraindicated.",
            source_document="Clinical Guideline for Pharyngitis",
            relevance_score=0.89,
        )
    ]
    retrieval_svc.retrieve.return_value = passages
    return retrieval_svc


@pytest.fixture
def configured_langgraph_engine(synthetic_extraction, synthetic_draft_soap, in_memory_retrieval_service):
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_llm_soap = MagicMock(spec=GeminiLLMService)
    mock_llm_soap.generate_structured.return_value = synthetic_draft_soap
    soap_svc = SOAPNoteService(llm_service=mock_llm_soap)

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=in_memory_retrieval_service,
        soap_service=soap_svc,
    )
    workflow = ClinicalWorkflow(services=services)
    return LangGraphClinicalEngine(workflow=workflow)


# =====================================================================
# Tests: 20 Phase 5 Requirements
# =====================================================================

def test_1_health_endpoint_still_works():
    """Requirement 1: Health endpoint returns 200 OK and valid health schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "meditrace-backend"
    assert "version" in data
    assert "timestamp" in data


def test_2_successful_consultation_invokes_injected_workflow(configured_langgraph_engine):
    """Requirement 2: Successful consultation invokes the injected LangGraph workflow."""
    app.dependency_overrides[get_clinical_engine] = lambda: configured_langgraph_engine
    try:
        payload = {
            "patient_id": "PT-LANGGRAPH-01",
            "patient_name": "Jane Anderson",
            "transcript": SYNTHETIC_TRANSCRIPT,
            "known_allergies": ["Penicillin"],
            "current_medications": [],
            "existing_conditions": [],
        }
        response = client.post("/api/v1/consultation", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["is_mock"] is False
        assert "LangGraph" in data["engine_source"]
        assert data["workflow_status"] == "completed"
        assert data["patient_id"] == "PT-LANGGRAPH-01"
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_3_real_engine_adapter_delegates_to_langgraph(configured_langgraph_engine):
    """Requirement 3: The real engine adapter (LangGraphClinicalEngine) delegates to LangGraph."""
    spy_workflow = MagicMock(wraps=configured_langgraph_engine.workflow)
    engine = LangGraphClinicalEngine(workflow=spy_workflow)
    app.dependency_overrides[get_clinical_engine] = lambda: engine

    try:
        payload = {
            "patient_id": "PT-SPY-01",
            "transcript": SYNTHETIC_TRANSCRIPT,
        }
        response = client.post("/api/v1/consultation", json=payload)
        assert response.status_code == 200
        spy_workflow.invoke.assert_called_once()
        call_arg = spy_workflow.invoke.call_args[0][0]
        assert call_arg["patient_id"] == "PT-SPY-01"
        assert call_arg["transcript"] == SYNTHETIC_TRANSCRIPT
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_4_soap_draft_fields_are_preserved(configured_langgraph_engine):
    """Requirement 4: SOAP draft fields (subjective, objective, assessment, plan) are preserved."""
    app.dependency_overrides[get_clinical_engine] = lambda: configured_langgraph_engine
    try:
        payload = {
            "patient_id": "PT-SOAP-FIELDS",
            "transcript": SYNTHETIC_TRANSCRIPT,
        }
        response = client.post("/api/v1/consultation", json=payload)
        assert response.status_code == 200
        draft = response.json()["draft_note"]
        assert "sore throat" in draft["subjective"].lower()
        assert "tonsillar exudate" in draft["objective"].lower()
        assert "tonsillopharyngitis" in draft["assessment"].lower()
        assert "Amoxicillin" in draft["plan"]
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_5_safety_flags_reach_api_response(configured_langgraph_engine):
    """Requirement 5: Deterministic safety review flags reach the API response in both fields."""
    app.dependency_overrides[get_clinical_engine] = lambda: configured_langgraph_engine
    try:
        payload = {
            "patient_id": "PT-SAFETY-ALERT",
            "transcript": SYNTHETIC_TRANSCRIPT,
            "known_allergies": ["Penicillin"],
        }
        response = client.post("/api/v1/consultation", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Both safety_flags and safety_alerts are populated
        assert len(data["safety_flags"]) >= 1
        assert len(data["safety_alerts"]) >= 1

        allergy_alerts = [
            a for a in data["safety_alerts"]
            if "penicillin" in a["title"].lower() or "amoxicillin" in a["description"].lower()
        ]
        assert len(allergy_alerts) >= 1
        assert allergy_alerts[0]["severity"] in ("MEDIUM", "HIGH", "CRITICAL")
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_6_missing_information_reaches_api_response(configured_langgraph_engine):
    """Requirement 6: Missing clinical information reaches the API response."""
    app.dependency_overrides[get_clinical_engine] = lambda: configured_langgraph_engine
    try:
        payload = {
            "patient_id": "PT-MISSING-INFO",
            "transcript": SYNTHETIC_TRANSCRIPT,
        }
        response = client.post("/api/v1/consultation", json=payload)
        assert response.status_code == 200
        missing = response.json()["missing_information"]
        assert len(missing) >= 1
        categories = [m["category"] for m in missing]
        assert "vital_signs" in categories or "allergy_history" in categories
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_7_reference_provenance_reaches_api_response(configured_langgraph_engine):
    """Requirement 7: Reference provenance metadata reaches the API response."""
    app.dependency_overrides[get_clinical_engine] = lambda: configured_langgraph_engine
    try:
        payload = {
            "patient_id": "PT-PROVENANCE",
            "transcript": SYNTHETIC_TRANSCRIPT,
        }
        response = client.post("/api/v1/consultation", json=payload)
        assert response.status_code == 200
        passages = response.json()["reference_passages"]
        assert len(passages) >= 1
        assert passages[0]["source_document"] == "Clinical Guideline for Pharyngitis"
        assert passages[0]["relevance_score"] >= 0.60
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_8_draft_status_and_disclaimer_preserved(configured_langgraph_engine):
    """Requirement 8: Output strictly preserves draft pending review status and safety disclaimer."""
    app.dependency_overrides[get_clinical_engine] = lambda: configured_langgraph_engine
    try:
        payload = {
            "patient_id": "PT-DISCLAIMER",
            "transcript": SYNTHETIC_TRANSCRIPT,
        }
        response = client.post("/api/v1/consultation", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["clinical_disclaimer"] == CLINICAL_SAFETY_DISCLAIMER
        assert data["draft_note"]["status"] == "draft_pending_review"
        assert "EDUCATIONAL PROTOTYPE ONLY" in data["draft_note"]["disclaimer"]
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_9_empty_and_whitespace_consultation_rejected(configured_langgraph_engine):
    """Requirement 9: Empty and whitespace consultation text are rejected (422 and 400)."""
    app.dependency_overrides[get_clinical_engine] = lambda: configured_langgraph_engine
    try:
        # Schema validation rejects under 10 chars with 422
        resp_too_short = client.post("/api/v1/consultation", json={"patient_id": "PT-SHORT", "transcript": "Short"})
        assert resp_too_short.status_code == 422

        # Whitespace-only string >= 10 chars is caught by engine validation and returns 400
        resp_whitespace = client.post(
            "/api/v1/consultation",
            json={"patient_id": "PT-SPACE", "transcript": "            "},
        )
        assert resp_whitespace.status_code == 400
        assert "non-whitespace" in resp_whitespace.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_10_extraction_failure_does_not_produce_successful_note():
    """Requirement 10: Clinical extraction failure halts workflow without producing a note (502)."""
    mock_llm_fail = MagicMock(spec=GeminiLLMService)
    mock_llm_fail.generate_structured.side_effect = ExtractionServiceError("LLM extraction failed")
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_fail)

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
    )
    workflow = ClinicalWorkflow(services=services)
    engine = LangGraphClinicalEngine(workflow=workflow)
    app.dependency_overrides[get_clinical_engine] = lambda: engine

    try:
        response = client.post("/api/v1/consultation", json={"patient_id": "PT-FAIL-01", "transcript": SYNTHETIC_TRANSCRIPT})
        assert response.status_code == 502
        assert "extraction failed" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_11_retrieval_no_evidence_distinguished_from_retrieval_failure(
    synthetic_extraction, synthetic_draft_soap
):
    """Requirement 11: Retrieval with no evidence is distinguished from retrieval failure."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_llm_soap = MagicMock(spec=GeminiLLMService)
    mock_llm_soap.generate_structured.return_value = synthetic_draft_soap
    soap_svc = SOAPNoteService(llm_service=mock_llm_soap)

    # 1. Retrieval returns no evidence (empty list)
    mock_retrieval_empty = MagicMock(spec=RetrievalService)
    mock_retrieval_empty.retrieve.return_value = []

    services_empty = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval_empty,
        soap_service=soap_svc,
    )
    app.dependency_overrides[get_clinical_engine] = lambda: LangGraphClinicalEngine(workflow=ClinicalWorkflow(services=services_empty))

    try:
        resp_empty = client.post("/api/v1/consultation", json={"patient_id": "PT-NO-EV", "transcript": SYNTHETIC_TRANSCRIPT})
        assert resp_empty.status_code == 200
        assert resp_empty.json()["retrieval_status"] == "no_evidence_found"
        assert len(resp_empty.json()["reference_passages"]) == 0

        # 2. Retrieval experiences a failure
        mock_retrieval_fail = MagicMock(spec=RetrievalService)
        mock_retrieval_fail.retrieve.side_effect = RetrievalServiceError("Vector store timeout")

        services_fail = ClinicalWorkflowServices(
            extraction_service=extraction_svc,
            retrieval_service=mock_retrieval_fail,
            soap_service=soap_svc,
        )
        app.dependency_overrides[get_clinical_engine] = lambda: LangGraphClinicalEngine(workflow=ClinicalWorkflow(services=services_fail))

        resp_fail = client.post("/api/v1/consultation", json={"patient_id": "PT-ERR-EV", "transcript": SYNTHETIC_TRANSCRIPT})
        assert resp_fail.status_code == 200
        assert resp_fail.json()["retrieval_status"] == "unavailable_error"
        assert len(resp_fail.json()["reference_passages"]) == 0
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_12_safety_verification_failure_does_not_silently_continue(synthetic_extraction):
    """Requirement 12: Safety verification internal failure halts workflow (500)."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    mock_safety_fail = MagicMock(spec=ClinicalSafetyVerificationService)
    mock_safety_fail.verify.side_effect = SafetyVerificationError("Safety rules corrupted")

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
        safety_service=mock_safety_fail,
    )
    workflow = ClinicalWorkflow(services=services)
    app.dependency_overrides[get_clinical_engine] = lambda: LangGraphClinicalEngine(workflow=workflow)

    try:
        response = client.post("/api/v1/consultation", json={"patient_id": "PT-SAFETY-FAIL", "transcript": SYNTHETIC_TRANSCRIPT})
        assert response.status_code == 500
        assert "safety verification failed" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_13_soap_generation_failure_produces_explicit_error(synthetic_extraction):
    """Requirement 13: SOAP generation failure produces explicit 502 error."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    mock_soap_fail = MagicMock(spec=SOAPNoteService)
    mock_soap_fail.generate_soap_note.side_effect = SOAPNoteGenerationError("LLM schema output validation failed")

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
        soap_service=mock_soap_fail,
    )
    workflow = ClinicalWorkflow(services=services)
    app.dependency_overrides[get_clinical_engine] = lambda: LangGraphClinicalEngine(workflow=workflow)

    try:
        response = client.post("/api/v1/consultation", json={"patient_id": "PT-SOAP-FAIL", "transcript": SYNTHETIC_TRANSCRIPT})
        assert response.status_code == 502
        assert "soap note generation failed" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_14_missing_live_configuration_produces_actionable_error(monkeypatch):
    """Requirement 14: Missing GEMINI_API_KEY in live mode produces actionable 503 error."""
    monkeypatch.setattr(settings, "mock_mode", False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = client.post("/api/v1/consultation", json={"patient_id": "PT-NO-KEY", "transcript": SYNTHETIC_TRANSCRIPT})
    assert response.status_code == 503
    assert "gemini_api_key" in response.json()["detail"].lower()


def test_15_mock_engine_used_only_when_explicitly_selected(monkeypatch):
    """Requirement 15: Mock engine is used only when explicitly selected via settings.mock_mode = True."""
    monkeypatch.setattr(settings, "mock_mode", True)
    engine = get_clinical_engine()
    assert isinstance(engine, MockClinicalEngine)

    response = client.post("/api/v1/consultation", json={"patient_id": "PT-MOCK-EXPLICIT", "transcript": SYNTHETIC_TRANSCRIPT})
    assert response.status_code == 200
    data = response.json()
    assert data["is_mock"] is True
    assert "MOCK" in data["engine_source"]
    assert "[MOCK RESPONSE]" in data["mock_disclaimer"]


def test_16_live_mode_never_silently_falls_back_to_mock_output(monkeypatch):
    """Requirement 16: Live mode never silently falls back to mock output on error."""
    monkeypatch.setattr(settings, "mock_mode", False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = client.post("/api/v1/consultation", json={"patient_id": "PT-NO-SILENT-FALLBACK", "transcript": SYNTHETIC_TRANSCRIPT})
    # Must fail with configuration error rather than returning 200 with mock note
    assert response.status_code == 503
    assert response.status_code != 200


def test_17_transcription_endpoint_remains_functional():
    """Requirement 17: Speech-to-text transcription endpoint remains functional."""
    app.dependency_overrides[get_transcription_engine] = lambda: MockTranscriptionEngine(
        simulated_transcript="Doctor: Tell me what brings you in. Patient: Sore throat for two days."
    )
    try:
        files = {"file": ("recording.wav", BytesIO(DUMMY_WAV_BYTES), "audio/wav")}
        response = client.post("/api/v1/transcribe", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "Sore throat" in data["transcript"]
        assert data["filename"] == "recording.wav"
    finally:
        app.dependency_overrides.pop(get_transcription_engine, None)


def test_18_mock_transcription_remains_test_only_and_clearly_identified():
    """Requirement 18: Mock transcription is clearly labeled with is_mock=True."""
    engine = MockTranscriptionEngine()
    resp = engine.transcribe(DUMMY_WAV_BYTES, "test.mp3")
    assert resp.is_mock is True
    assert "MOCK" in resp.engine_name


def test_19_no_live_gemini_calls_occur_during_tests(monkeypatch, configured_langgraph_engine):
    """Requirement 19: Offline test execution makes zero live external network connections."""
    def guarded_connect(*args, **kwargs):
        raise AssertionError("Live network socket connection attempted during unit testing!")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)

    app.dependency_overrides[get_clinical_engine] = lambda: configured_langgraph_engine
    try:
        response = client.post(
            "/api/v1/consultation",
            json={"patient_id": "PT-OFFLINE-TEST", "transcript": SYNTHETIC_TRANSCRIPT},
        )
        assert response.status_code == 200
        assert response.json()["workflow_status"] == "completed"
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)


def test_20_no_api_key_required_for_offline_test_suite(monkeypatch, configured_langgraph_engine):
    """Requirement 20: Offline test suite with injected workflow runs without GEMINI_API_KEY."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert os.getenv("GEMINI_API_KEY") is None

    app.dependency_overrides[get_clinical_engine] = lambda: configured_langgraph_engine
    try:
        response = client.post(
            "/api/v1/consultation",
            json={"patient_id": "PT-NO-KEY-INJECTED", "transcript": SYNTHETIC_TRANSCRIPT},
        )
        assert response.status_code == 200
        assert response.json()["is_mock"] is False
    finally:
        app.dependency_overrides.pop(get_clinical_engine, None)
