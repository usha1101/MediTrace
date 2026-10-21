"""Unit tests for the speech-to-text (STT) transcription endpoint.

Tests verify validation, error handling, mock engine behavior, and full audio-to-consultation flow
WITHOUT downloading speech models or requiring physical microphones.
"""

from io import BytesIO
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.transcription import (
    MockTranscriptionEngine,
    TranscriptionEngineInterface,
    TranscriptionEngineUnavailableError,
    TranscriptionError,
    get_transcription_engine,
)

client = TestClient(app)

# Dummy non-empty WAV header for testing
DUMMY_WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def test_transcribe_success_mock(monkeypatch):
    """Verify that a valid audio file returns 200 OK and expected transcript."""
    # Ensure mock engine is used for isolated, instant testing
    app.dependency_overrides[get_transcription_engine] = lambda: MockTranscriptionEngine(
        simulated_transcript="Doctor: Hello. Patient: I have had a cough for 3 days."
    )
    try:
        files = {
            "file": ("encounter_recording.wav", BytesIO(DUMMY_WAV_BYTES), "audio/wav")
        }
        response = client.post("/api/v1/transcribe", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data["filename"] == "encounter_recording.wav"
        assert "Doctor: Hello. Patient: I have had a cough for 3 days." in data["transcript"]
        assert data["is_mock"] is True
        assert "MOCK" in data["engine_name"]
        assert "created_at" in data
    finally:
        app.dependency_overrides.pop(get_transcription_engine, None)


def test_transcribe_empty_file():
    """Verify that uploading a 0-byte file returns 400 Bad Request."""
    files = {
        "file": ("empty.wav", BytesIO(b""), "audio/wav")
    }
    response = client.post("/api/v1/transcribe", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcribe_invalid_file_type():
    """Verify that uploading an unsupported file extension returns 415 Unsupported Media Type."""
    files = {
        "file": ("notes.txt", BytesIO(b"Doctor notes in plain text"), "text/plain")
    }
    response = client.post("/api/v1/transcribe", files=files)
    assert response.status_code == 415
    assert "unsupported" in response.json()["detail"].lower()


def test_transcribe_file_too_large(monkeypatch):
    """Verify that files exceeding the maximum size limit return 413 Payload Too Large."""
    from backend.config import settings
    # Temporarily set limit to 100 bytes for fast testing
    original_limit = settings.max_audio_size_bytes
    settings.max_audio_size_bytes = 100

    try:
        oversized_bytes = b"0" * 150
        files = {
            "file": ("large_recording.wav", BytesIO(oversized_bytes), "audio/wav")
        }
        response = client.post("/api/v1/transcribe", files=files)
        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"].lower()
    finally:
        settings.max_audio_size_bytes = original_limit


def test_transcribe_engine_error():
    """Verify that internal transcription errors return HTTP 500."""
    class FailingEngine(TranscriptionEngineInterface):
        def transcribe(self, audio_bytes: bytes, filename: str):
            raise TranscriptionError("Simulated audio decoder failure")

    app.dependency_overrides[get_transcription_engine] = lambda: FailingEngine()
    try:
        files = {
            "file": ("test.wav", BytesIO(DUMMY_WAV_BYTES), "audio/wav")
        }
        response = client.post("/api/v1/transcribe", files=files)
        assert response.status_code == 500
        assert "transcription processing error" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_transcription_engine, None)


def test_transcribe_engine_unavailable():
    """Verify that missing model weights or engine unavailability returns HTTP 503."""
    class UnavailableEngine(TranscriptionEngineInterface):
        def transcribe(self, audio_bytes: bytes, filename: str):
            raise TranscriptionEngineUnavailableError("Model weights not loaded")

    app.dependency_overrides[get_transcription_engine] = lambda: UnavailableEngine()
    try:
        files = {
            "file": ("test.wav", BytesIO(DUMMY_WAV_BYTES), "audio/wav")
        }
        response = client.post("/api/v1/transcribe", files=files)
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_transcription_engine, None)


def test_full_audio_to_consultation_pipeline():
    """Verify complete end-to-end integration: audio upload -> transcription -> consultation note."""
    app.dependency_overrides[get_transcription_engine] = lambda: MockTranscriptionEngine(
        simulated_transcript=(
            "Doctor: Good morning Jane. Let me examine your throat. "
            "It looks like bacterial pharyngitis. I will prescribe Amoxicillin 500mg. "
            "Patient: Thank you doctor."
        )
    )

    try:
        # Step 1: Transcribe audio
        files = {
            "file": ("patient_visit.mp3", BytesIO(DUMMY_WAV_BYTES), "audio/mpeg")
        }
        stt_resp = client.post("/api/v1/transcribe", files=files)
        assert stt_resp.status_code == 200
        transcribed_text = stt_resp.json()["transcript"]
        assert "Amoxicillin" in transcribed_text

        # Step 2: Feed transcribed text into consultation endpoint
        consult_payload = {
            "patient_id": "PT-AUDIO-99",
            "patient_name": "Jane Doe",
            "transcript": transcribed_text,
            "known_allergies": ["Penicillin"],
            "current_medications": [],
            "existing_conditions": [],
        }
        consult_resp = client.post("/api/v1/consultation", json=consult_payload)
        assert consult_resp.status_code == 200

        consult_data = consult_resp.json()
        assert consult_data["is_mock"] is True
        assert consult_data["patient_id"] == "PT-AUDIO-99"

        # Verify draft SOAP note and penicillin allergy conflict are generated from transcribed text
        assert "pharyngitis" in consult_data["draft_note"]["assessment"].lower()
        high_alerts = [a for a in consult_data["safety_alerts"] if a["severity"] == "HIGH"]
        assert len(high_alerts) >= 1
        assert "Penicillin" in high_alerts[0]["title"]
    finally:
        app.dependency_overrides.pop(get_transcription_engine, None)
