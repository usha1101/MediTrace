"""
Unit and Integration Tests for MediTrace Frontend API Client and Integration Logic.

Tests Phase 6 Streamlit frontend integration:
- backend health checks (online, offline, timeout, malformed)
- successful text consultation
- backend connection failure, timeouts, HTTP errors, malformed JSON
- input validation (patient ID, transcript length)
- mock vs live response identification
- extraction and normalization of SOAP fields, safety flags, missing info, references
- retrieval status categorization
- audio transcription (success, mock flag, errors, empty bytes)
- end-to-end transcript submission into clinical consultation workflow

All tests run 100% offline using mocked network requests.
Zero live Gemini or external network calls.
"""

import json
from unittest.mock import MagicMock, patch
import pytest
import requests

from frontend.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiHttpError,
    ApiPayloadError,
    ApiTimeoutError,
    check_backend_health,
    extract_missing_information,
    extract_references,
    extract_safety_flags,
    extract_soap_fields,
    get_default_backend_url,
    get_retrieval_status_info,
    is_mock_response,
    is_mock_transcription,
    submit_consultation,
    submit_transcription,
)


# =====================================================================
# 1. Backend URL Configuration Tests
# =====================================================================

def test_get_default_backend_url_fallback(monkeypatch):
    """Test default URL fallback to local development port 8000."""
    monkeypatch.delenv("MEDITRACE_BACKEND_URL", raising=False)
    monkeypatch.delenv("FASTAPI_BACKEND_URL", raising=False)
    assert get_default_backend_url() == "http://127.0.0.1:8000"


def test_get_default_backend_url_from_meditrace_env(monkeypatch):
    """Test URL configuration via MEDITRACE_BACKEND_URL."""
    monkeypatch.setenv("MEDITRACE_BACKEND_URL", "http://meditrace.internal:9000/")
    assert get_default_backend_url() == "http://meditrace.internal:9000"


def test_get_default_backend_url_from_fastapi_env(monkeypatch):
    """Test URL configuration fallback to FASTAPI_BACKEND_URL."""
    monkeypatch.delenv("MEDITRACE_BACKEND_URL", raising=False)
    monkeypatch.setenv("FASTAPI_BACKEND_URL", "https://api.meditrace.org:8443")
    assert get_default_backend_url() == "https://api.meditrace.org:8443"


# =====================================================================
# 2. Health Check Service Tests
# =====================================================================

@patch("requests.get")
def test_check_backend_health_online(mock_get):
    """Test successful health check against online FastAPI backend."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "healthy",
        "service": "meditrace-backend",
        "version": "0.1.0",
        "mock_engine_active": False,
        "timestamp": "2026-10-21T12:00:00Z",
    }
    mock_get.return_value = mock_resp

    result = check_backend_health("http://127.0.0.1:8000")
    assert result["online"] is True
    assert result["version"] == "0.1.0"
    assert result["mock_engine_active"] is False
    assert result["error"] is None
    assert result["data"]["service"] == "meditrace-backend"


@patch("requests.get")
def test_check_backend_health_connection_error(mock_get):
    """Test health check when backend is offline / connection refused."""
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

    result = check_backend_health("http://127.0.0.1:8000")
    assert result["online"] is False
    assert "Could not connect" in result["error"]
    assert result["data"] is None


@patch("requests.get")
def test_check_backend_health_timeout(mock_get):
    """Test health check when request times out."""
    mock_get.side_effect = requests.exceptions.Timeout("Read timed out")

    result = check_backend_health("http://127.0.0.1:8000", timeout=2.0)
    assert result["online"] is False
    assert "timed out" in result["error"]


@patch("requests.get")
def test_check_backend_health_non_200(mock_get):
    """Test health check when backend returns non-200 status."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_get.return_value = mock_resp

    result = check_backend_health("http://127.0.0.1:8000")
    assert result["online"] is False
    assert "HTTP 503" in result["error"]


@patch("requests.get")
def test_check_backend_health_malformed_json(mock_get):
    """Test health check when backend payload is not valid JSON."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Invalid JSON")
    mock_get.return_value = mock_resp

    result = check_backend_health("http://127.0.0.1:8000")
    assert result["online"] is False
    assert "Invalid health response payload" in result["error"]


# =====================================================================
# 3. Consultation Service Tests
# =====================================================================

@pytest.fixture
def sample_consultation_payload():
    return {
        "patient_id": "PT-9402",
        "patient_name": "Jane Anderson",
        "patient_age": 34,
        "patient_gender": "Female",
        "transcript": "Doctor: Good morning Jane. Patient: I have had a severe sore throat and fever for 2 days.",
        "known_allergies": ["Penicillin"],
        "current_medications": ["Cetirizine 10mg"],
        "existing_conditions": ["Seasonal allergic rhinitis"],
    }


@pytest.fixture
def sample_backend_consultation_response():
    return {
        "consultation_id": "cons-test-1234",
        "patient_id": "PT-9402",
        "encounter_id": "enc-PT-9402-abc",
        "is_mock": False,
        "engine_source": "LangGraphClinicalEngine (Gemini 2.5 Flash + Qdrant RAG + Deterministic Safety)",
        "mock_disclaimer": None,
        "clinical_disclaimer": "EDUCATIONAL PROTOTYPE ONLY: Requires licensed clinician verification.",
        "workflow_status": "completed",
        "retrieval_status": "success",
        "draft_note": {
            "note_id": "note-001",
            "subjective": "Patient reports 2-day history of sore throat and fever.",
            "objective": "Tonsillar erythema observed. Vitals not recorded.",
            "assessment": "Acute pharyngitis (suspected upper respiratory infection).",
            "plan": "Hold antibiotic pending rapid strep test. Note penicillin allergy.",
            "status": "draft_pending_review",
            "disclaimer": "EDUCATIONAL PROTOTYPE ONLY: Requires licensed clinician verification.",
        },
        "safety_flags": [
            {
                "flag_id": "FLAG-ALLERGY-01",
                "category": "drug_allergy_conflict",
                "severity": "high",
                "title": "Contraindicated Antibiotic Alert",
                "detail": "Patient has documented penicillin allergy; beta-lactams contraindicated.",
                "flagged_items": ["Penicillin", "Amoxicillin"],
                "requires_clinician_acknowledgment": True,
                "clinician_review_status": "pending_review",
            }
        ],
        "missing_information": [
            {
                "category": "vital_signs",
                "description": "Temperature and blood pressure were not measured or documented.",
                "clinical_relevance": "Objective vitals are required to assess systemic toxicity.",
                "recommended_follow_up": "Measure core body temperature and vital signs before discharge.",
            }
        ],
        "reference_passages": [
            {
                "passage_id": "ref-nice-ng28-1",
                "title": "NICE NG28 - Sore Throat (Acute)",
                "content": "Do not offer an antibiotic for acute sore throat without Centor score evaluation.",
                "source_document": "NICE Guideline NG28",
                "section_or_page": "Section 1.1",
                "relevance_score": 0.89,
                "retrieval_metadata": {"document_id": "doc-ng28"},
            }
        ],
        "suggested_icd10_codes": ["J02.9 - Acute pharyngitis, unspecified"],
        "created_at": "2026-10-21T12:00:00Z",
    }


@patch("requests.post")
def test_submit_consultation_success(mock_post, sample_consultation_payload, sample_backend_consultation_response):
    """Test successful submission of consultation payload to backend."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = sample_backend_consultation_response
    mock_post.return_value = mock_resp

    data = submit_consultation(sample_consultation_payload, base_url="http://127.0.0.1:8000")

    assert data["consultation_id"] == "cons-test-1234"
    assert data["is_mock"] is False
    assert data["workflow_status"] == "completed"
    assert "draft_note" in data
    assert data["draft_note"]["subjective"] == "Patient reports 2-day history of sore throat and fever."

    # Verify requests.post called with correct endpoint and timeout
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == "http://127.0.0.1:8000/api/v1/consultation"
    assert call_kwargs["json"] == sample_consultation_payload
    assert call_kwargs["timeout"] > 0


@patch("requests.post")
def test_submit_consultation_connection_failure(mock_post, sample_consultation_payload):
    """Test connection failure raises ApiConnectionError without falling back to mock."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused by host")

    with pytest.raises(ApiConnectionError) as exc_info:
        submit_consultation(sample_consultation_payload, base_url="http://127.0.0.1:8000")

    assert "Could not connect to backend server" in str(exc_info.value)


@patch("requests.post")
def test_submit_consultation_timeout(mock_post, sample_consultation_payload):
    """Test request timeout raises ApiTimeoutError."""
    mock_post.side_effect = requests.exceptions.Timeout("Read timeout")

    with pytest.raises(ApiTimeoutError) as exc_info:
        submit_consultation(sample_consultation_payload, base_url="http://127.0.0.1:8000", timeout=10.0)

    assert "timed out after 10.0s" in str(exc_info.value)


@patch("requests.post")
def test_submit_consultation_http_error_400(mock_post, sample_consultation_payload):
    """Test HTTP 400 Bad Request raises ApiHttpError with backend detail."""
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"detail": "Transcript contains insufficient dialogue."}
    mock_post.return_value = mock_resp

    with pytest.raises(ApiHttpError) as exc_info:
        submit_consultation(sample_consultation_payload, base_url="http://127.0.0.1:8000")

    assert exc_info.value.status_code == 400
    assert "Transcript contains insufficient dialogue" in exc_info.value.detail


@patch("requests.post")
def test_submit_consultation_http_error_502(mock_post, sample_consultation_payload):
    """Test HTTP 502 Bad Gateway from clinical workflow failure."""
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.json.return_value = {"detail": "Clinical workflow execution error: extraction failed."}
    mock_post.return_value = mock_resp

    with pytest.raises(ApiHttpError) as exc_info:
        submit_consultation(sample_consultation_payload, base_url="http://127.0.0.1:8000")

    assert exc_info.value.status_code == 502
    assert "Clinical workflow execution error" in exc_info.value.detail


@patch("requests.post")
def test_submit_consultation_malformed_json_response(mock_post, sample_consultation_payload):
    """Test malformed JSON response raises ApiPayloadError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    mock_post.return_value = mock_resp

    with pytest.raises(ApiPayloadError) as exc_info:
        submit_consultation(sample_consultation_payload, base_url="http://127.0.0.1:8000")

    assert "malformed JSON response" in str(exc_info.value)


def test_submit_consultation_client_validation_errors():
    """Test client-side validation errors before network call is attempted."""
    # 1. Missing patient_id
    with pytest.raises(ApiClientError) as exc1:
        submit_consultation({"patient_id": "", "transcript": "Doctor: Hello world dialogue."})
    assert "Patient ID is required" in str(exc1.value)

    # 2. Too short transcript
    with pytest.raises(ApiClientError) as exc2:
        submit_consultation({"patient_id": "PT-100", "transcript": "Too short"})
    assert "at least 10 non-whitespace characters" in str(exc2.value)

    # 3. Non-dict payload
    with pytest.raises(ApiClientError) as exc3:
        submit_consultation("not-a-dict")  # type: ignore
    assert "must be a dictionary" in str(exc3.value)


# =====================================================================
# 4. Mock Engine & Live Workflow Indicator Tests
# =====================================================================

def test_mock_response_clearly_identified():
    """Verify mock vs live indicators are accurately extracted."""
    live_response = {
        "is_mock": False,
        "engine_source": "LangGraphClinicalEngine (Gemini 2.5 Flash + RAG)",
    }
    mock_response = {
        "is_mock": True,
        "engine_source": "MOCK_CLINICAL_AI_ENGINE (Awaiting Member A Model)",
        "mock_disclaimer": "[MOCK RESPONSE] Prototype output.",
    }

    assert is_mock_response(live_response) is False
    assert is_mock_response(mock_response) is True


def test_mock_transcription_clearly_identified():
    """Verify mock speech recognition flag is accurately extracted."""
    live_stt = {"engine_name": "faster-whisper-base.en", "is_mock": False}
    mock_stt = {"engine_name": "MockTranscriptionEngine", "is_mock": True}

    assert is_mock_transcription(live_stt) is False
    assert is_mock_transcription(mock_stt) is True


# =====================================================================
# 5. Extraction & Normalization Helper Tests
# =====================================================================

def test_extract_soap_fields_complete(sample_backend_consultation_response):
    """Test extracting all SOAP sections and metadata."""
    soap = extract_soap_fields(sample_backend_consultation_response)
    assert soap["subjective"] == "Patient reports 2-day history of sore throat and fever."
    assert soap["objective"] == "Tonsillar erythema observed. Vitals not recorded."
    assert soap["assessment"] == "Acute pharyngitis (suspected upper respiratory infection)."
    assert soap["plan"] == "Hold antibiotic pending rapid strep test. Note penicillin allergy."
    assert soap["status"] == "draft_pending_review"
    assert soap["note_id"] == "note-001"
    assert "EDUCATIONAL PROTOTYPE" in soap["disclaimer"]


def test_extract_soap_fields_empty_graceful():
    """Test graceful handling of empty or missing SOAP note."""
    soap = extract_soap_fields({})
    assert soap["subjective"] == ""
    assert soap["objective"] == ""
    assert soap["assessment"] == ""
    assert soap["plan"] == ""
    assert soap["status"] == "draft_pending_review"
    assert soap["note_id"] is None


def test_extract_safety_flags_structured(sample_backend_consultation_response):
    """Test extracting structured safety flags from LangGraph engine."""
    flags = extract_safety_flags(sample_backend_consultation_response)
    assert len(flags) == 1
    flag = flags[0]
    assert flag["flag_id"] == "FLAG-ALLERGY-01"
    assert flag["severity"] == "HIGH"
    assert flag["category"] == "drug_allergy_conflict"
    assert flag["title"] == "Contraindicated Antibiotic Alert"
    assert "beta-lactams contraindicated" in flag["detail"]
    assert "Amoxicillin" in flag["flagged_items"]
    assert flag["requires_acknowledgment"] is True
    assert flag["review_status"] == "pending_review"


def test_extract_safety_flags_from_mock_alerts():
    """Test fallback to safety_alerts from MockClinicalEngine."""
    mock_resp = {
        "safety_alerts": [
            {
                "alert_id": "alert-001",
                "severity": "HIGH",
                "category": "Allergy Contraindication",
                "title": "Penicillin Allergy Alert",
                "description": "Amoxicillin proposed despite documented penicillin allergy.",
                "recommendation": "Do not prescribe Amoxicillin.",
            }
        ],
        "clinical_disclaimer": "Safety notice",
    }
    flags = extract_safety_flags(mock_resp)
    assert len(flags) == 1
    flag = flags[0]
    assert flag["flag_id"] == "alert-001"
    assert flag["severity"] == "HIGH"
    assert flag["title"] == "Penicillin Allergy Alert"
    assert flag["recommendation"] == "Do not prescribe Amoxicillin."


def test_extract_safety_flags_empty():
    """Test extraction when no safety alerts or flags exist."""
    assert extract_safety_flags({}) == []


def test_extract_missing_information(sample_backend_consultation_response):
    """Test extracting identified missing information items."""
    missing = extract_missing_information(sample_backend_consultation_response)
    assert len(missing) == 1
    item = missing[0]
    assert item["category"] == "vital_signs"
    assert "Temperature and blood pressure" in item["description"]
    assert "Objective vitals are required" in item["clinical_relevance"]
    assert "Measure core body temperature" in item["recommended_follow_up"]


def test_extract_missing_information_empty():
    """Test extraction when no missing information is reported."""
    assert extract_missing_information({}) == []


def test_extract_references(sample_backend_consultation_response):
    """Test extracting guideline references with provenance."""
    refs = extract_references(sample_backend_consultation_response)
    assert len(refs) == 1
    ref = refs[0]
    assert ref["passage_id"] == "ref-nice-ng28-1"
    assert ref["source_document"] == "NICE Guideline NG28"
    assert ref["title"] == "NICE NG28 - Sore Throat (Acute)"
    assert ref["section_or_page"] == "Section 1.1"
    assert ref["relevance_score"] == 0.89
    assert "Do not offer an antibiotic" in ref["content"]
    assert ref["retrieval_metadata"]["document_id"] == "doc-ng28"


def test_extract_references_empty():
    """Test extraction when no references were retrieved."""
    assert extract_references({}) == []


def test_get_retrieval_status_info_variants():
    """Test mapping retrieval status strings into UI badge specifications."""
    # 1. Success
    res_success = get_retrieval_status_info("success")
    assert res_success["badge_type"] == "success"
    assert res_success["label"] == "Evidence Retrieved"

    # 2. No evidence found
    res_no_ev = get_retrieval_status_info("no_evidence_found")
    assert res_no_ev["badge_type"] == "warning"
    assert res_no_ev["label"] == "No Evidence Found"

    # 3. Unavailable / error
    res_err = get_retrieval_status_info("unavailable_error")
    assert res_err["badge_type"] == "danger"
    assert res_err["label"] == "Retrieval Unavailable"

    # 4. None / not reported
    res_none = get_retrieval_status_info(None)
    assert res_none["badge_type"] == "neutral"
    assert res_none["label"] == "Retrieval Not Performed"


# =====================================================================
# 6. Audio Transcription Service Tests
# =====================================================================

@pytest.fixture
def sample_audio_bytes():
    # 44-byte standard empty WAV header
    return b"RIFF,\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"


@patch("requests.post")
def test_submit_transcription_success(mock_post, sample_audio_bytes):
    """Test successful transcription request to FastAPI backend."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "filename": "encounter_recording.wav",
        "transcript": "Doctor: Hello. Patient: My throat has been hurting for two days.",
        "language": "en",
        "duration_seconds": 4.5,
        "engine_name": "faster-whisper-base.en",
        "is_mock": False,
        "created_at": "2026-10-21T12:00:00Z",
    }
    mock_post.return_value = mock_resp

    data = submit_transcription(
        audio_bytes=sample_audio_bytes,
        filename="encounter_recording.wav",
        base_url="http://127.0.0.1:8000",
    )

    assert data["filename"] == "encounter_recording.wav"
    assert data["transcript"] == "Doctor: Hello. Patient: My throat has been hurting for two days."
    assert data["is_mock"] is False
    assert data["engine_name"] == "faster-whisper-base.en"

    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == "http://127.0.0.1:8000/api/v1/transcribe"
    assert "file" in call_kwargs["files"]


@patch("requests.post")
def test_submit_transcription_mock_engine(mock_post, sample_audio_bytes):
    """Test transcription clearly labeled as mock output."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "filename": "recording.wav",
        "transcript": "[MOCK AUDIO TRANSCRIPT] Doctor: Hello.",
        "language": "en",
        "duration_seconds": 3.0,
        "engine_name": "MockTranscriptionEngine",
        "is_mock": True,
        "created_at": "2026-10-21T12:00:00Z",
    }
    mock_post.return_value = mock_resp

    data = submit_transcription(sample_audio_bytes, "recording.wav")
    assert is_mock_transcription(data) is True
    assert data["engine_name"] == "MockTranscriptionEngine"


@patch("requests.post")
def test_submit_transcription_connection_error(mock_post, sample_audio_bytes):
    """Test transcription connection failure raises ApiConnectionError."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

    with pytest.raises(ApiConnectionError) as exc_info:
        submit_transcription(sample_audio_bytes, "recording.wav")

    assert "Could not connect to backend server" in str(exc_info.value)


@patch("requests.post")
def test_submit_transcription_timeout(mock_post, sample_audio_bytes):
    """Test transcription timeout raises ApiTimeoutError."""
    mock_post.side_effect = requests.exceptions.Timeout("Processing timed out")

    with pytest.raises(ApiTimeoutError) as exc_info:
        submit_transcription(sample_audio_bytes, "recording.wav", timeout=15.0)

    assert "timed out after 15.0s" in str(exc_info.value)


@patch("requests.post")
def test_submit_transcription_http_415(mock_post, sample_audio_bytes):
    """Test unsupported audio file format raises ApiHttpError with 415."""
    mock_resp = MagicMock()
    mock_resp.status_code = 415
    mock_resp.json.return_value = {"detail": "Unsupported audio format '.txt'."}
    mock_post.return_value = mock_resp

    with pytest.raises(ApiHttpError) as exc_info:
        submit_transcription(sample_audio_bytes, "notes.txt")

    assert exc_info.value.status_code == 415
    assert "Unsupported audio format" in exc_info.value.detail


def test_submit_transcription_empty_bytes_client_validation():
    """Test empty audio bytes raises ApiClientError without HTTP request."""
    with pytest.raises(ApiClientError) as exc_info:
        submit_transcription(b"", "recording.wav")
    assert "Cannot transcribe empty audio buffer" in str(exc_info.value)


# =====================================================================
# 7. End-to-End Workflow: Audio Transcript into Consultation
# =====================================================================

@patch("requests.post")
def test_audio_transcript_submission_through_consultation_workflow(
    mock_post,
    sample_audio_bytes,
    sample_backend_consultation_response,
):
    """
    Test complete Phase 6 integration loop:
    1. Transcribe audio to obtain transcript.
    2. Review and pass the transcribed dialogue into consultation payload.
    3. Submit to consultation workflow and receive structured note with safety verification.
    """
    # Step 1: Mock transcription response
    mock_stt_resp = MagicMock()
    mock_stt_resp.status_code = 200
    mock_stt_resp.json.return_value = {
        "filename": "visit.wav",
        "transcript": "Doctor: Jane has tonsillar exudate. Prescribing Amoxicillin 500mg.",
        "language": "en",
        "duration_seconds": 5.2,
        "engine_name": "faster-whisper-base.en",
        "is_mock": False,
        "created_at": "2026-10-21T12:00:00Z",
    }

    # Step 2: Mock consultation response
    mock_cons_resp = MagicMock()
    mock_cons_resp.status_code = 200
    mock_cons_resp.json.return_value = sample_backend_consultation_response

    # Alternate between transcription and consultation calls
    mock_post.side_effect = [mock_stt_resp, mock_cons_resp]

    # Run STT
    stt_result = submit_transcription(sample_audio_bytes, "visit.wav")
    assert stt_result["transcript"] == "Doctor: Jane has tonsillar exudate. Prescribing Amoxicillin 500mg."
    assert is_mock_transcription(stt_result) is False

    # Review/edit transcript (simulated clinician workflow step)
    reviewed_transcript = stt_result["transcript"] + " Patient has penicillin allergy."

    # Build consultation payload
    consultation_payload = {
        "patient_id": "PT-9402",
        "transcript": reviewed_transcript,
        "known_allergies": ["Penicillin"],
        "current_medications": [],
        "existing_conditions": [],
    }

    # Run Consultation
    cons_result = submit_consultation(consultation_payload)

    # Validate output
    assert cons_result["consultation_id"] == "cons-test-1234"
    assert cons_result["workflow_status"] == "completed"

    soap = extract_soap_fields(cons_result)
    assert soap["status"] == "draft_pending_review"
    assert len(soap["subjective"]) > 0

    flags = extract_safety_flags(cons_result)
    assert len(flags) >= 1
    assert any(f["severity"] == "HIGH" for f in flags)

    refs = extract_references(cons_result)
    assert len(refs) >= 1
    assert refs[0]["source_document"] == "NICE Guideline NG28"

    assert is_mock_response(cons_result) is False
