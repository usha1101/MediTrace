"""
MediTrace - Frontend API Client Service.

Provides testable, robust HTTP communication between the Streamlit UI
and the FastAPI clinical copilot backend.
Handles request validation, timeouts, connection errors, HTTP errors,
and response serialization without leaking secrets.
"""

import logging
import os
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

# Default timeouts in seconds
DEFAULT_HEALTH_TIMEOUT = 3.0
DEFAULT_CONSULTATION_TIMEOUT = 45.0
DEFAULT_TRANSCRIBE_TIMEOUT = 120.0


# =====================================================================
# API Client Exceptions
# =====================================================================

class ApiClientError(Exception):
    """Base exception for MediTrace frontend API client errors."""
    pass


class ApiConnectionError(ApiClientError):
    """Raised when connection to backend server cannot be established."""
    pass


class ApiTimeoutError(ApiClientError):
    """Raised when backend request times out."""
    pass


class ApiHttpError(ApiClientError):
    """Raised when backend returns an error HTTP status code."""

    def __init__(self, status_code: int, detail: str, response_text: Optional[str] = None):
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail
        self.response_text = response_text or ""


class ApiPayloadError(ApiClientError):
    """Raised when backend response payload is malformed or invalid JSON."""
    pass


# =====================================================================
# Configuration Helper
# =====================================================================

def get_default_backend_url() -> str:
    """
    Resolve the default FastAPI backend base URL.
    Checks environment variables MEDITRACE_BACKEND_URL and FASTAPI_BACKEND_URL,
    falling back to standard local development URL http://127.0.0.1:8000.
    """
    env_url = os.getenv("MEDITRACE_BACKEND_URL") or os.getenv("FASTAPI_BACKEND_URL")
    if env_url and env_url.strip():
        return env_url.strip().rstrip("/")
    return "http://127.0.0.1:8000"


# =====================================================================
# Health Check Service
# =====================================================================

def check_backend_health(
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_HEALTH_TIMEOUT,
) -> Dict[str, Any]:
    """
    Check availability and operational readiness of the FastAPI backend.

    Args:
        base_url: Base URL of backend service. Defaults to get_default_backend_url().
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys:
            - online: bool
            - version: str (if online)
            - mock_engine_active: bool (if online)
            - error: str (if offline)
            - data: dict (raw health response if online)
    """
    url = (base_url or get_default_backend_url()).rstrip("/")
    endpoint = f"{url}/health"

    try:
        response = requests.get(endpoint, timeout=timeout)
        if response.status_code == 200:
            try:
                data = response.json()
                return {
                    "online": True,
                    "version": data.get("version", "0.1.0"),
                    "mock_engine_active": data.get("mock_engine_active", False),
                    "service": data.get("service", "meditrace-backend"),
                    "data": data,
                    "error": None,
                }
            except Exception as json_err:
                return {
                    "online": False,
                    "error": f"Invalid health response payload: {str(json_err)}",
                    "data": None,
                }
        return {
            "online": False,
            "error": f"Backend returned HTTP {response.status_code}",
            "data": None,
        }
    except requests.exceptions.Timeout:
        return {
            "online": False,
            "error": f"Connection timed out after {timeout} seconds",
            "data": None,
        }
    except requests.exceptions.ConnectionError as conn_err:
        return {
            "online": False,
            "error": f"Could not connect to {url}: {str(conn_err)}",
            "data": None,
        }
    except Exception as exc:
        return {
            "online": False,
            "error": f"Unexpected health check error: {str(exc)}",
            "data": None,
        }


# =====================================================================
# Consultation Service
# =====================================================================

def submit_consultation(
    payload: Dict[str, Any],
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_CONSULTATION_TIMEOUT,
) -> Dict[str, Any]:
    """
    Submit clinical encounter dialogue and patient metadata to the FastAPI backend.

    Args:
        payload: Dict conforming to ConsultationRequest schema:
            - patient_id: str (required)
            - transcript: str (required, min 10 chars)
            - patient_name: Optional[str]
            - patient_age: Optional[int]
            - patient_gender: Optional[str]
            - known_allergies: List[str]
            - current_medications: List[str]
            - existing_conditions: List[str]
        base_url: Base URL of backend service.
        timeout: Request timeout in seconds.

    Returns:
        Dict conforming to ConsultationResponse schema.

    Raises:
        ApiConnectionError: If network connection fails.
        ApiTimeoutError: If request times out.
        ApiHttpError: If backend returns an error status code.
        ApiPayloadError: If response body cannot be decoded.
    """
    url = (base_url or get_default_backend_url()).rstrip("/")
    endpoint = f"{url}/api/v1/consultation"

    if not isinstance(payload, dict):
        raise ApiClientError("Consultation payload must be a dictionary.")

    patient_id = payload.get("patient_id", "")
    transcript = payload.get("transcript", "")

    if not str(patient_id).strip():
        raise ApiClientError("Patient ID is required.")

    if not str(transcript).strip() or len(str(transcript).strip()) < 10:
        raise ApiClientError("Consultation transcript must be at least 10 non-whitespace characters.")

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise ApiTimeoutError(f"Consultation request timed out after {timeout}s.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise ApiConnectionError(f"Could not connect to backend server at {url}.") from exc
    except Exception as exc:
        raise ApiClientError(f"Request failed: {str(exc)}") from exc

    # Parse response body
    if response.status_code != 200:
        detail = "Clinical engine processing failed"
        try:
            err_json = response.json()
            if isinstance(err_json, dict) and "detail" in err_json:
                detail = str(err_json["detail"])
        except Exception:
            detail = response.text or detail

        raise ApiHttpError(
            status_code=response.status_code,
            detail=detail,
            response_text=response.text,
        )

    try:
        data = response.json()
    except Exception as exc:
        raise ApiPayloadError(f"Backend returned malformed JSON response: {str(exc)}") from exc

    if not isinstance(data, dict):
        raise ApiPayloadError("Backend returned invalid response structure (expected dictionary).")

    return data


# =====================================================================
# Audio Transcription Service
# =====================================================================

def submit_transcription(
    audio_bytes: bytes,
    filename: str,
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_TRANSCRIBE_TIMEOUT,
) -> Dict[str, Any]:
    """
    Submit audio recording bytes to the speech-to-text transcription service.

    Args:
        audio_bytes: Raw bytes of uploaded audio file.
        filename: Original filename (e.g., 'visit.wav').
        base_url: Base URL of backend service.
        timeout: Request timeout in seconds.

    Returns:
        Dict conforming to TranscriptionResponse schema:
            - filename: str
            - transcript: str
            - language: str
            - duration_seconds: float
            - engine_name: str
            - is_mock: bool
            - created_at: str

    Raises:
        ApiConnectionError: If network connection fails.
        ApiTimeoutError: If transcription times out.
        ApiHttpError: If backend returns error status.
        ApiPayloadError: If response body cannot be decoded.
    """
    url = (base_url or get_default_backend_url()).rstrip("/")
    endpoint = f"{url}/api/v1/transcribe"

    if not audio_bytes or len(audio_bytes) == 0:
        raise ApiClientError("Cannot transcribe empty audio buffer (0 bytes).")

    if not filename or not filename.strip():
        raise ApiClientError("Audio filename must be specified.")

    files = {"file": (filename, audio_bytes, "application/octet-stream")}

    try:
        response = requests.post(
            endpoint,
            files=files,
            timeout=timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise ApiTimeoutError(f"Transcription request timed out after {timeout}s.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise ApiConnectionError(f"Could not connect to backend server at {url}.") from exc
    except Exception as exc:
        raise ApiClientError(f"Request failed: {str(exc)}") from exc

    if response.status_code != 200:
        detail = "Audio transcription failed"
        try:
            err_json = response.json()
            if isinstance(err_json, dict) and "detail" in err_json:
                detail = str(err_json["detail"])
        except Exception:
            detail = response.text or detail

        raise ApiHttpError(
            status_code=response.status_code,
            detail=detail,
            response_text=response.text,
        )

    try:
        data = response.json()
    except Exception as exc:
        raise ApiPayloadError(f"Backend returned malformed JSON response: {str(exc)}") from exc

    return data


# =====================================================================
# Response Extraction & Normalization Helpers
# =====================================================================

def is_mock_response(response: Dict[str, Any]) -> bool:
    """Return True if the consultation response originated from a mock engine."""
    return bool(response.get("is_mock", False))


def is_mock_transcription(response: Dict[str, Any]) -> bool:
    """Return True if the audio transcription originated from a mock engine."""
    return bool(response.get("is_mock", False))


def extract_soap_fields(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract SOAP note sections, note identifier, and lifecycle status
    from a consultation response dict.
    """
    draft_note = response.get("draft_note") or {}
    return {
        "subjective": draft_note.get("subjective", ""),
        "objective": draft_note.get("objective", ""),
        "assessment": draft_note.get("assessment", ""),
        "plan": draft_note.get("plan", ""),
        "note_id": draft_note.get("note_id"),
        "status": draft_note.get("status") or "draft_pending_review",
        "disclaimer": draft_note.get("disclaimer") or response.get("clinical_disclaimer") or "",
    }


def extract_safety_flags(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and normalize structured clinical safety review flags from consultation response.
    Supports both Member A `safety_flags` and Member B `safety_alerts` schemas.
    """
    normalized: List[Dict[str, Any]] = []

    # 1. Prefer structured safety_flags (from Member A / LangGraph engine)
    flags = response.get("safety_flags") or []
    if not flags:
        # Check inside draft_note if not top-level
        flags = (response.get("draft_note") or {}).get("safety_flags") or []

    if flags:
        for f in flags:
            if not isinstance(f, dict):
                continue
            sev = str(f.get("severity", "INFO")).upper()
            if hasattr(f.get("severity"), "value"):
                sev = str(f["severity"].value).upper()
            cat = f.get("category", "Clinical Safety")
            if hasattr(f.get("category"), "value"):
                cat = str(f["category"].value)
            normalized.append({
                "flag_id": f.get("flag_id", f.get("alert_id", "FLAG-GENERIC")),
                "severity": sev,
                "category": cat,
                "title": f.get("title", "Clinical Safety Warning"),
                "detail": f.get("detail") or f.get("description", ""),
                "recommendation": f.get("recommendation") or f.get("detail", "Clinician review required."),
                "flagged_items": f.get("flagged_items") or [],
                "supporting_references": f.get("supporting_references") or [],
                "requires_acknowledgment": f.get("requires_clinician_acknowledgment", True),
                "review_status": f.get("clinician_review_status", "pending_review"),
                "disclaimer": f.get("disclaimer") or response.get("clinical_disclaimer") or "",
            })
        return normalized

    # 2. Fall back to safety_alerts (from Member B MockClinicalEngine)
    alerts = response.get("safety_alerts") or []
    for a in alerts:
        if not isinstance(a, dict):
            continue
        sev = str(a.get("severity", "INFO")).upper()
        normalized.append({
            "flag_id": a.get("alert_id", "ALERT-GENERIC"),
            "severity": sev,
            "category": a.get("category", "Clinical Safety"),
            "title": a.get("title", "Clinical Alert"),
            "detail": a.get("description", ""),
            "recommendation": a.get("recommendation", "Clinician review required."),
            "flagged_items": [],
            "supporting_references": [],
            "requires_acknowledgment": True,
            "review_status": "pending_review",
            "disclaimer": response.get("clinical_disclaimer") or "",
        })

    return normalized


def extract_missing_information(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract missing clinical information items from consultation response.
    """
    missing = response.get("missing_information") or []
    if not missing:
        missing = (response.get("draft_note") or {}).get("missing_information") or []

    normalized = []
    for item in missing:
        if isinstance(item, dict):
            normalized.append({
                "category": item.get("category", "General Clinical Data"),
                "description": item.get("description", ""),
                "clinical_relevance": item.get("clinical_relevance", ""),
                "recommended_follow_up": item.get("recommended_follow_up"),
            })
    return normalized


def extract_references(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract retrieved guideline passages and provenance metadata.
    """
    refs = response.get("reference_passages") or []
    if not refs:
        refs = (response.get("draft_note") or {}).get("references") or []

    normalized = []
    for r in refs:
        if isinstance(r, dict):
            normalized.append({
                "passage_id": r.get("passage_id", ""),
                "title": r.get("title", "Guideline Reference"),
                "content": r.get("content", ""),
                "source_document": r.get("source_document", "Clinical Guideline"),
                "source_url_or_doi": r.get("source_url_or_doi"),
                "section_or_page": r.get("section_or_page"),
                "relevance_score": r.get("relevance_score"),
                "retrieval_metadata": r.get("retrieval_metadata") or {},
            })
    return normalized


def get_retrieval_status_info(status_str: Optional[str]) -> Dict[str, Any]:
    """
    Provide human-readable label and UI styling for retrieval status.
    """
    if not status_str:
        return {
            "status": "not_reported",
            "label": "Retrieval Not Performed",
            "badge_type": "neutral",
            "description": "Guideline retrieval status was not reported by the backend.",
        }

    status_lower = status_str.lower().strip()
    if status_lower == "success":
        return {
            "status": "success",
            "label": "Evidence Retrieved",
            "badge_type": "success",
            "description": "Clinical guideline evidence successfully retrieved from knowledge base.",
        }
    elif status_lower == "no_evidence_found":
        return {
            "status": "no_evidence_found",
            "label": "No Evidence Found",
            "badge_type": "warning",
            "description": "No guideline passages in the knowledge base met the relevance threshold.",
        }
    elif "unavailable" in status_lower or "error" in status_lower:
        return {
            "status": "error",
            "label": "Retrieval Unavailable",
            "badge_type": "danger",
            "description": "Vector store or embedding service was unavailable during consultation processing.",
        }
    else:
        return {
            "status": status_lower,
            "label": status_str,
            "badge_type": "info",
            "description": f"Retrieval status: {status_str}",
        }
