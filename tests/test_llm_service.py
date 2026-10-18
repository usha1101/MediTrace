"""
Unit tests for GeminiLLMService (app/services/llm_service.py).
Mocks all Gemini API calls so tests run without real API keys or network access.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors
from pydantic import BaseModel, Field

from app.schemas import (
    CLINICAL_SAFETY_DISCLAIMER,
    AllergyInfo,
    CertaintyLevel,
    ClinicalExtractionResult,
    InformationSource,
    MedicationMention,
    MedicationStatus,
    PresenceStatus,
    SymptomInfo,
)
from app.services.llm_service import (
    DEFAULT_GEMINI_MODEL,
    GeminiLLMService,
    LLMAPIError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMResponseParsingError,
    LLMValidationError,
)


# Simple test schema for generic structured testing
class SimpleClinicalEntity(BaseModel):
    category: str
    item_name: str
    confidence: float = Field(ge=0.0, le=1.0)


# =====================================================================
# 1. Initialization and Configuration Tests
# =====================================================================

def test_init_with_explicit_api_key():
    mock_client = MagicMock()
    service = GeminiLLMService(api_key="test-api-key-12345", client=mock_client)
    assert service.model == DEFAULT_GEMINI_MODEL
    assert service.client is mock_client


def test_init_with_env_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-api-key-xyz")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    mock_client = MagicMock()

    with patch("google.genai.Client", return_value=mock_client) as mock_client_cls:
        service = GeminiLLMService()
        assert service.model == DEFAULT_GEMINI_MODEL
        mock_client_cls.assert_called_once_with(api_key="env-api-key-xyz")


def test_init_raises_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError) as exc_info:
        GeminiLLMService()
    assert "GEMINI_API_KEY is not set or empty" in str(exc_info.value)


def test_init_raises_error_when_api_key_empty():
    with pytest.raises(LLMConfigurationError) as exc_info:
        GeminiLLMService(api_key="   ")
    assert "GEMINI_API_KEY is not set or empty" in str(exc_info.value)


def test_custom_model_via_parameter():
    mock_client = MagicMock()
    service = GeminiLLMService(api_key="test-key", model="gemini-2.0-pro", client=mock_client)
    assert service.model == "gemini-2.0-pro"


def test_custom_model_via_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash-custom")
    mock_client = MagicMock()
    service = GeminiLLMService(client=mock_client)
    assert service.model == "gemini-2.0-flash-custom"


# =====================================================================
# 2. Successful Structured Output Generation
# =====================================================================

def test_generate_structured_success_simple_model():
    mock_client = MagicMock()
    fake_response = MagicMock()
    fake_response.text = json.dumps({
        "category": "symptom",
        "item_name": "headache",
        "confidence": 0.95,
    })
    mock_client.models.generate_content.return_value = fake_response

    service = GeminiLLMService(api_key="test-key", client=mock_client)
    result = service.generate_structured(
        prompt="Extract entity from: patient has mild headache",
        response_model=SimpleClinicalEntity,
        system_instruction="Extract clinical findings.",
    )

    assert isinstance(result, SimpleClinicalEntity)
    assert result.category == "symptom"
    assert result.item_name == "headache"
    assert result.confidence == 0.95

    # Verify parameters passed to SDK
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_GEMINI_MODEL
    assert call_kwargs["contents"] == "Extract entity from: patient has mild headache"

    config = call_kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is SimpleClinicalEntity
    assert config.system_instruction == "Extract clinical findings."
    assert config.temperature == 0.0


def test_generate_structured_success_clinical_extraction_result():
    mock_client = MagicMock()
    fake_response = MagicMock()

    # Synthetic consultation output
    extraction_payload = {
        "encounter_id": "enc_synth_001",
        "symptoms": [
            {
                "name": "Dry cough",
                "status": "present",
                "certainty": "uncertain",
                "source": "patient_reported",
                "duration": "4 days",
                "raw_quote": "I have had a dry cough for about 4 days",
            }
        ],
        "allergies": [
            {
                "allergen": "Penicillin",
                "reaction": "Rash",
                "source": "patient_reported",
                "verified_by_clinician": False,
            }
        ],
        "medications": [
            {
                "name": "Metformin",
                "dosage": "500 mg",
                "status": "current",
                "source": "patient_reported",
            }
        ],
        "missing_information": [
            {
                "category": "vitals",
                "description": "Temperature not measured",
                "clinical_relevance": "Need to assess fever for acute cough",
            }
        ],
        "consultation_summary": "Patient presented with a 4-day history of dry cough.",
        "disclaimer": CLINICAL_SAFETY_DISCLAIMER,
    }
    fake_response.text = json.dumps(extraction_payload)
    mock_client.models.generate_content.return_value = fake_response

    service = GeminiLLMService(api_key="test-key", client=mock_client)
    result = service.generate_structured(
        prompt="Doctor-Patient Dialogue: ...",
        response_model=ClinicalExtractionResult,
    )

    assert isinstance(result, ClinicalExtractionResult)
    assert result.encounter_id == "enc_synth_001"
    assert len(result.symptoms) == 1
    assert result.symptoms[0].name == "Dry cough"
    assert result.symptoms[0].certainty == CertaintyLevel.UNCERTAIN
    assert len(result.allergies) == 1
    assert result.allergies[0].allergen == "Penicillin"
    assert result.allergies[0].verified_by_clinician is False
    assert len(result.medications) == 1
    assert result.medications[0].status == MedicationStatus.CURRENT
    assert len(result.missing_information) == 1
    assert CLINICAL_SAFETY_DISCLAIMER in result.disclaimer


def test_generate_structured_strips_markdown_code_fences():
    mock_client = MagicMock()
    fake_response = MagicMock()
    # Model wrapped the JSON in markdown code blocks
    fake_response.text = "```json\n{\"category\": \"vital\", \"item_name\": \"heart_rate\", \"confidence\": 0.88}\n```"
    mock_client.models.generate_content.return_value = fake_response

    service = GeminiLLMService(api_key="test-key", client=mock_client)
    result = service.generate_structured(
        prompt="Extract finding",
        response_model=SimpleClinicalEntity,
    )
    assert isinstance(result, SimpleClinicalEntity)
    assert result.item_name == "heart_rate"


# =====================================================================
# 3. Error Handling and Edge Cases Tests
# =====================================================================

def test_generate_structured_rejects_empty_prompt():
    mock_client = MagicMock()
    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMConfigurationError) as exc_info:
        service.generate_structured(prompt="", response_model=SimpleClinicalEntity)
    assert "Prompt must not be empty" in str(exc_info.value)

    with pytest.raises(LLMConfigurationError) as exc_info:
        service.generate_structured(prompt="   \n\t  ", response_model=SimpleClinicalEntity)
    assert "Prompt must not be empty" in str(exc_info.value)


def test_generate_structured_rejects_non_base_model():
    mock_client = MagicMock()
    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMConfigurationError) as exc_info:
        service.generate_structured(prompt="Hello", response_model=dict)  # type: ignore
    assert "response_model must be a subclass of pydantic.BaseModel" in str(exc_info.value)


def test_generate_structured_handles_gemini_api_error():
    mock_client = MagicMock()
    api_error = errors.APIError(code=503, response_json={"message": "Service Unavailable"})
    mock_client.models.generate_content.side_effect = api_error

    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMAPIError) as exc_info:
        service.generate_structured(prompt="Analyze transcript", response_model=SimpleClinicalEntity)
    assert "Gemini API error" in str(exc_info.value)


def test_generate_structured_handles_generic_network_error():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ConnectionResetError("Connection dropped")

    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMAPIError) as exc_info:
        service.generate_structured(prompt="Analyze transcript", response_model=SimpleClinicalEntity)
    assert "Unexpected error communicating with Gemini API" in str(exc_info.value)


def test_generate_structured_handles_none_response():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = None

    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMEmptyResponseError) as exc_info:
        service.generate_structured(prompt="Analyze transcript", response_model=SimpleClinicalEntity)
    assert "empty response with no content" in str(exc_info.value)


def test_generate_structured_handles_empty_text():
    mock_client = MagicMock()
    fake_response = MagicMock()
    fake_response.text = None
    mock_client.models.generate_content.return_value = fake_response

    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMEmptyResponseError) as exc_info:
        service.generate_structured(prompt="Analyze transcript", response_model=SimpleClinicalEntity)
    assert "empty response with no content" in str(exc_info.value)


def test_generate_structured_handles_whitespace_response():
    mock_client = MagicMock()
    fake_response = MagicMock()
    fake_response.text = "   \n\t   "
    mock_client.models.generate_content.return_value = fake_response

    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMEmptyResponseError) as exc_info:
        service.generate_structured(prompt="Analyze transcript", response_model=SimpleClinicalEntity)
    assert "whitespace-only response" in str(exc_info.value)


def test_generate_structured_handles_malformed_json():
    mock_client = MagicMock()
    fake_response = MagicMock()
    fake_response.text = "This is plain conversational text without JSON brackets"
    mock_client.models.generate_content.return_value = fake_response

    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMResponseParsingError) as exc_info:
        service.generate_structured(prompt="Analyze transcript", response_model=SimpleClinicalEntity)
    assert "Failed to decode response as valid JSON" in str(exc_info.value)


def test_generate_structured_handles_pydantic_validation_error():
    mock_client = MagicMock()
    fake_response = MagicMock()
    # Missing required 'category' and 'item_name' fields
    fake_response.text = json.dumps({"confidence": 1.5})  # also invalid confidence > 1.0
    mock_client.models.generate_content.return_value = fake_response

    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMValidationError) as exc_info:
        service.generate_structured(prompt="Analyze transcript", response_model=SimpleClinicalEntity)
    assert "Response failed Pydantic validation" in str(exc_info.value)


def test_generate_structured_never_fabricates_data_on_failure():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Fatal LLM crash")

    service = GeminiLLMService(api_key="test-key", client=mock_client)

    with pytest.raises(LLMAPIError):
        # Must raise exception, never return empty/dummy ClinicalExtractionResult
        service.generate_structured(prompt="Analyze dialogue", response_model=ClinicalExtractionResult)
