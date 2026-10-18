"""
MediTrace - Clinical AI Engine LLM Service
Reusable integration with the official google-genai Python SDK.

CLINICAL SAFETY NOTICE:
-----------------------
MediTrace is an educational clinical documentation prototype.
It must NOT autonomously diagnose, prescribe, or make treatment decisions.
All LLM-generated clinical summaries, draft documentation, and safety review
flags require review and verification by a qualified, licensed clinician.
The absence of an alert or flag must NEVER be interpreted as proof that a patient is safe.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Type, TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# Configurable default Gemini model: current fast, multimodal, structured-JSON capable model
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"

T = TypeVar("T", bound=BaseModel)


# =====================================================================
# Service Exceptions
# =====================================================================

class LLMServiceError(Exception):
    """Base exception for all MediTrace LLM service operations."""
    pass


class LLMConfigurationError(LLMServiceError):
    """Raised when environment variables or required service configurations are missing or invalid."""
    pass


class LLMAPIError(LLMServiceError):
    """Raised when the underlying Gemini API call fails."""
    pass


class LLMEmptyResponseError(LLMServiceError):
    """Raised when the Gemini API returns an empty, null, or whitespace-only response."""
    pass


class LLMResponseParsingError(LLMServiceError):
    """Raised when the Gemini API response is not valid JSON."""
    pass


class LLMValidationError(LLMServiceError):
    """Raised when the parsed JSON response fails Pydantic schema validation."""
    pass


# =====================================================================
# Gemini LLM Service Implementation
# =====================================================================

class GeminiLLMService:
    """
    Reusable Gemini LLM client supporting validated Pydantic v2 structured outputs.

    Design Principles:
    - Independent of web frameworks (no FastAPI/Streamlit dependencies).
    - Reads GEMINI_API_KEY from environment or explicit parameter.
    - Reads GEMINI_MODEL from environment (defaults to DEFAULT_GEMINI_MODEL).
    - Never logs API keys or sensitive clinical consultation text.
    - Strictly validates responses against requested Pydantic models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[genai.Client] = None,
    ):
        """
        Initialize the Gemini LLM service.

        Args:
            api_key: Optional API key override. If not provided, reads from GEMINI_API_KEY env var.
            model: Optional model name override. If not provided, reads from GEMINI_MODEL env var.
            client: Optional pre-configured genai.Client instance (for dependency injection/mocking).

        Raises:
            LLMConfigurationError: If no API key is provided or found in the environment.
        """
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not set or empty. Please configure the GEMINI_API_KEY "
                "environment variable or pass an explicit api_key."
            )
        self._api_key: str = resolved_key.strip()

        resolved_model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        if not resolved_model or not resolved_model.strip():
            resolved_model = DEFAULT_GEMINI_MODEL
        self.model: str = resolved_model.strip()

        if client is not None:
            self._client = client
        else:
            try:
                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:
                # Do not log or expose the API key in the error message
                raise LLMConfigurationError(
                    f"Failed to initialize google-genai Client: {type(e).__name__}"
                ) from e

    @property
    def client(self) -> genai.Client:
        """Access the underlying google-genai Client instance."""
        return self._client

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> T:
        """
        Generate structured output from the Gemini API and validate it against a Pydantic model.

        Args:
            prompt: User prompt content.
            response_model: Pydantic v2 model class to constrain and validate the response against.
            system_instruction: Optional system instruction for role definition and formatting rules.
            temperature: Sampling temperature (default 0.0 for deterministic extraction/safety).

        Returns:
            An instance of response_model populated with validated data from the LLM.

        Raises:
            LLMConfigurationError: If prompt or response_model is invalid.
            LLMAPIError: If the Gemini API call encounters a network, authentication, or service error.
            LLMEmptyResponseError: If the model returns no content.
            LLMResponseParsingError: If the returned text is not valid JSON.
            LLMValidationError: If the JSON cannot be validated against response_model.
        """
        if not prompt or not prompt.strip():
            raise LLMConfigurationError("Prompt must not be empty or whitespace only.")

        if not isinstance(response_model, type) or not issubclass(response_model, BaseModel):
            raise LLMConfigurationError(
                f"response_model must be a subclass of pydantic.BaseModel, got {response_model}."
            )

        # Configure structured output using google-genai SDK
        config_kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": response_model,
            "temperature": temperature,
        }
        if system_instruction and system_instruction.strip():
            config_kwargs["system_instruction"] = system_instruction.strip()

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            # Notice: Never log the raw prompt or API key to prevent exposing consultation data
            logger.info(
                "Calling Gemini API (model: %s) for structured schema %s",
                self.model,
                response_model.__name__,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except errors.APIError as e:
            # Sanitized error handling: do not expose consultation text, URL query params, or secrets
            error_code = getattr(e, "code", None)
            code_str = f" (HTTP {error_code})" if error_code is not None else ""
            error_msg = getattr(e, "message", None) or type(e).__name__
            logger.error("Gemini API error during generation%s", code_str)
            raise LLMAPIError(f"Gemini API error{code_str}: {error_msg}") from e
        except Exception as e:
            logger.error("Unexpected error during Gemini API call: %s", type(e).__name__)
            raise LLMAPIError(f"Unexpected error communicating with Gemini API: {type(e).__name__}") from e

        # Validate response presence
        if response is None or not hasattr(response, "text") or response.text is None:
            raise LLMEmptyResponseError("Gemini API returned an empty response with no content.")

        raw_text = response.text.strip()
        if not raw_text:
            raise LLMEmptyResponseError("Gemini API returned a whitespace-only response.")

        # Clean potential markdown fences (e.g., ```json ... ```)
        cleaned_text = self._strip_markdown_code_fences(raw_text)

        # 1. Parse cleaned_text using json.loads()
        # 2. Catch json.JSONDecodeError and raise LLMResponseParsingError
        try:
            parsed_json = json.loads(cleaned_text)
        except json.JSONDecodeError as json_err:
            logger.error("Failed to parse Gemini response as JSON for model %s", response_model.__name__)
            raise LLMResponseParsingError(
                f"Failed to decode response as valid JSON for {response_model.__name__}: {json_err.msg} (line {json_err.lineno}, col {json_err.colno})."
            ) from json_err

        # 3. Validate the parsed Python object using response_model.model_validate()
        # 4. Catch Pydantic ValidationError and raise LLMValidationError
        try:
            validated_instance = response_model.model_validate(parsed_json)
            return validated_instance
        except ValidationError as val_err:
            logger.error("Pydantic validation failed for %s", response_model.__name__)
            raise LLMValidationError(
                f"Response failed Pydantic validation for {response_model.__name__}: {val_err.error_count()} error(s) detected."
            ) from val_err

    @staticmethod
    def _strip_markdown_code_fences(text: str) -> str:
        """
        Remove markdown code fences (```json ... ```) if the model wrapped output in them.
        """
        trimmed = text.strip()
        if trimmed.startswith("```"):
            first_line_end = trimmed.find("\n")
            if first_line_end != -1:
                trimmed = trimmed[first_line_end + 1 :]
            else:
                trimmed = trimmed.lstrip("`")
            if trimmed.endswith("```"):
                trimmed = trimmed[:-3].rstrip()
        return trimmed
