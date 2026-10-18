"""
Unit tests for MediTrace Embedding Service (app/services/embedding_service.py).

Tests:
- Model configuration (defaults, parameter overrides, env var overrides, rejection of invalid models)
- Vector dimensionality (default 768, custom output_dimensionality, dimension mismatch validation)
- Batch size configuration (positive integer validation, rejection of boolean values)
- Successful single-text and batch embedding with input order preservation
- Empty and invalid input handling
- Invalid provider response handling (None, missing embeddings, missing values)
- Empty and non-finite vector handling (NaN, +inf, -inf, non-numeric)
- API error sanitization, HTTP status categorization, and secret/transcript leakage prevention
- DocumentChunk integration and provenance metadata preservation
- Log sanitization (ensuring secrets and patient text are not logged)
"""

import math
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors

from app.schemas import DocumentChunk
from app.services.embedding_service import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    ChunkEmbedding,
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    EmbeddingService,
    EmbeddingValidationError,
)


# =====================================================================
# Test Helpers
# =====================================================================

def make_vector(dim=DEFAULT_EMBEDDING_DIMENSION, start_val=0.1):
    """Generate a valid test vector of exact length `dim`."""
    return [(start_val + i * 0.001) for i in range(dim)]


def make_mock_embedding(values):
    """Create a mock ContentEmbedding object with .values attribute."""
    mock_emb = MagicMock()
    mock_emb.values = values
    return mock_emb


def make_mock_response(embeddings_values_list):
    """Create a mock EmbedContentResponse with .embeddings list."""
    mock_resp = MagicMock()
    mock_resp.embeddings = [make_mock_embedding(vals) for vals in embeddings_values_list]
    return mock_resp


# =====================================================================
# 1. Initialization, Model & Configuration Tests
# =====================================================================

def test_init_defaults():
    mock_client = MagicMock()
    service = EmbeddingService(api_key="test-api-key-12345", client=mock_client)
    assert DEFAULT_EMBEDDING_MODEL == "gemini-embedding-001"
    assert service.model == "gemini-embedding-001"
    assert service.output_dimensionality == DEFAULT_EMBEDDING_DIMENSION
    assert service.output_dimensionality == 768
    assert service.batch_size == 100
    assert service.client is mock_client



def test_init_with_env_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-api-key-xyz")
    monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_EMBEDDING_DIMENSION", raising=False)
    mock_client = MagicMock()

    with patch("google.genai.Client", return_value=mock_client):
        service = EmbeddingService()
        assert service.model == DEFAULT_EMBEDDING_MODEL
        assert service.output_dimensionality == DEFAULT_EMBEDDING_DIMENSION
        assert service.client is mock_client


def test_init_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EmbeddingConfigurationError, match="GEMINI_API_KEY is not set or empty"):
        EmbeddingService()


@pytest.mark.parametrize("empty_key", ["", "   ", None])
def test_init_empty_api_key_raises_error(empty_key, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EmbeddingConfigurationError, match="GEMINI_API_KEY is not set or empty"):
        EmbeddingService(api_key=empty_key)


def test_init_custom_model_parameter():
    mock_client = MagicMock()
    service = EmbeddingService(
        api_key="key",
        model="text-embedding-004",
        client=mock_client,
    )
    assert service.model == "text-embedding-004"


def test_init_custom_model_via_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "custom-embedding-model")
    mock_client = MagicMock()

    service = EmbeddingService(client=mock_client)
    assert service.model == "custom-embedding-model"


@pytest.mark.parametrize("invalid_model", ["", "   ", 12345])
def test_init_rejects_invalid_model_parameter(invalid_model):
    mock_client = MagicMock()
    with pytest.raises(EmbeddingConfigurationError, match="Embedding model parameter must be a non-empty string"):
        EmbeddingService(api_key="key", client=mock_client, model=invalid_model)


@pytest.mark.parametrize("invalid_env_model", ["", "   "])
def test_init_rejects_empty_env_model(invalid_env_model, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", invalid_env_model)
    mock_client = MagicMock()

    with pytest.raises(EmbeddingConfigurationError, match="GEMINI_EMBEDDING_MODEL environment variable must not be empty"):
        EmbeddingService(client=mock_client)


# =====================================================================
# 2. Batch Size Validation & Boolean Rejection Tests
# =====================================================================

@pytest.mark.parametrize(
    "invalid_batch_size",
    [
        0,
        -1,
        -100,
        True,   # Critical check: bool must not be accepted as int
        False,  # Critical check: bool must not be accepted as int
        "100",
        None,
        1.5,
    ],
)
def test_init_rejects_invalid_or_boolean_batch_size(invalid_batch_size):
    mock_client = MagicMock()
    with pytest.raises(EmbeddingConfigurationError, match="batch_size must be a positive integer"):
        EmbeddingService(api_key="key", client=mock_client, batch_size=invalid_batch_size)


# =====================================================================
# 3. Vector Dimensionality Configuration Tests
# =====================================================================

def test_init_custom_output_dimensionality_parameter():
    mock_client = MagicMock()
    service = EmbeddingService(
        api_key="key",
        client=mock_client,
        output_dimensionality=512,
    )
    assert service.output_dimensionality == 512


def test_init_custom_output_dimensionality_via_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSION", "256")
    mock_client = MagicMock()

    service = EmbeddingService(client=mock_client)
    assert service.output_dimensionality == 256


@pytest.mark.parametrize(
    "invalid_dim",
    [
        0,
        -1,
        -768,
        True,
        False,
        "invalid",
        None,
    ],
)
def test_init_rejects_invalid_output_dimensionality_param(invalid_dim):
    if invalid_dim is None:
        return  # None is valid for defaulting to DEFAULT_EMBEDDING_DIMENSION
    mock_client = MagicMock()
    with pytest.raises(EmbeddingConfigurationError, match="output_dimensionality must be a positive integer"):
        EmbeddingService(api_key="key", client=mock_client, output_dimensionality=invalid_dim)


@pytest.mark.parametrize("invalid_env_dim", ["0", "-5", "not_a_number", ""])
def test_init_rejects_invalid_env_dimension(invalid_env_dim, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSION", invalid_env_dim)
    mock_client = MagicMock()

    with pytest.raises(EmbeddingConfigurationError, match="GEMINI_EMBEDDING_DIMENSION must be a positive integer"):
        EmbeddingService(client=mock_client)


def test_init_client_failure_raises_sanitized_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("google.genai.Client", side_effect=RuntimeError("SDK init boom")):
        with pytest.raises(EmbeddingConfigurationError, match="Failed to initialize google-genai Client"):
            EmbeddingService(api_key="secret-key-12345")


# =====================================================================
# 4. Vector Validation Tests
# =====================================================================

def test_validate_vector_valid():
    vec = [0.1, -0.25, 0.0, 100.5]
    result = EmbeddingService.validate_vector(vec)
    assert result == vec
    assert all(isinstance(x, float) for x in result)


def test_validate_vector_with_expected_dim():
    vec = [1.0, 2.0, 3.0]
    assert EmbeddingService.validate_vector(vec, expected_dim=3) == vec

    with pytest.raises(EmbeddingValidationError, match="dimension mismatch"):
        EmbeddingService.validate_vector(vec, expected_dim=4)


@pytest.mark.parametrize("empty_vec", [[], ()])
def test_validate_vector_rejects_empty(empty_vec):
    with pytest.raises(EmbeddingValidationError, match="must not be empty"):
        EmbeddingService.validate_vector(empty_vec)


@pytest.mark.parametrize("non_seq", ["not-a-list", 123, None, {"a": 1}])
def test_validate_vector_rejects_non_sequence(non_seq):
    with pytest.raises(EmbeddingValidationError, match="must be a sequence of numbers"):
        EmbeddingService.validate_vector(non_seq)


@pytest.mark.parametrize(
    "invalid_values",
    [
        [0.1, float("nan"), 0.3],
        [float("inf"), 0.2, 0.3],
        [0.1, float("-inf"), 0.3],
        [0.1, "not-a-number", 0.3],
        [0.1, None, 0.3],
        [0.1, True, 0.3],  # bool is subclass of int in Python, must be caught
    ],
)
def test_validate_vector_rejects_non_finite_or_non_numeric(invalid_values):
    with pytest.raises(EmbeddingValidationError, match="(non-finite|non-numeric)"):
        EmbeddingService.validate_vector(invalid_values)


# =====================================================================
# 5. Single-Text & Query Embedding Tests
# =====================================================================

def test_embed_single_text_success_default_dimension():
    mock_client = MagicMock()
    expected_vec = make_vector(768)
    mock_client.models.embed_content.return_value = make_mock_response([expected_vec])

    service = EmbeddingService(api_key="test-key", client=mock_client)
    result = service.embed_text("Clinical guidelines for community-acquired pneumonia.")

    assert len(result) == 768
    assert result == expected_vec
    mock_client.models.embed_content.assert_called_once()
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_EMBEDDING_MODEL
    assert call_kwargs["contents"] == "Clinical guidelines for community-acquired pneumonia."
    assert call_kwargs["config"].output_dimensionality == 768


def test_embed_single_text_with_custom_dimensionality():
    mock_client = MagicMock()
    custom_dim = 256
    expected_vec = make_vector(custom_dim)
    mock_client.models.embed_content.return_value = make_mock_response([expected_vec])

    service = EmbeddingService(
        api_key="test-key",
        client=mock_client,
        output_dimensionality=custom_dim,
    )
    result = service.embed_text("Asthma exacerbation clinical protocol.")

    assert len(result) == custom_dim
    assert result == expected_vec
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["config"].output_dimensionality == custom_dim


def test_dimension_mismatch_raises_validation_error():
    mock_client = MagicMock()
    # Service expects 768, but provider returns 512
    wrong_dim_vec = make_vector(512)
    mock_client.models.embed_content.return_value = make_mock_response([wrong_dim_vec])

    service = EmbeddingService(api_key="test-key", client=mock_client)  # default 768
    with pytest.raises(EmbeddingValidationError, match="Embedding vector dimension mismatch: expected 768, got 512"):
        service.embed_text("Sample reference text")


def test_embed_query_sets_retrieval_query_task_type():
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = make_mock_response([make_vector(768)])

    service = EmbeddingService(api_key="test-key", client=mock_client)
    res = service.embed_query("amoxicillin pediatric dosage")

    assert len(res) == 768
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["config"].task_type == "RETRIEVAL_QUERY"
    assert call_kwargs["config"].output_dimensionality == 768


@pytest.mark.parametrize("empty_input", ["", "   ", "\n\t  ", None, 123])
def test_embed_text_rejects_empty_or_invalid_input(empty_input):
    mock_client = MagicMock()
    service = EmbeddingService(api_key="test-key", client=mock_client)

    with pytest.raises(EmbeddingConfigurationError, match="Text to embed must not be empty"):
        service.embed_text(empty_input)


# =====================================================================
# 6. Batch Embedding & Order Preservation Tests
# =====================================================================

def test_embed_texts_preserves_input_order():
    mock_client = MagicMock()
    texts = [
        "First guideline on Type 2 diabetes monotherapy.",
        "Second guideline on SGLT2 inhibitor dual therapy.",
        "Third guideline on lifestyle and nutrition interventions.",
    ]
    vectors = [
        make_vector(128, start_val=1.0),
        make_vector(128, start_val=2.0),
        make_vector(128, start_val=3.0),
    ]
    mock_client.models.embed_content.return_value = make_mock_response(vectors)

    service = EmbeddingService(
        api_key="test-key",
        client=mock_client,
        output_dimensionality=128,
    )
    results = service.embed_texts(texts)

    assert len(results) == 3
    # Strict order preservation check
    assert results[0] == vectors[0]
    assert results[1] == vectors[1]
    assert results[2] == vectors[2]


def test_embed_texts_batching_multiple_calls():
    mock_client = MagicMock()
    texts = [f"Text {i}" for i in range(5)]
    dim = 64
    vecs = [make_vector(dim, start_val=float(i)) for i in range(5)]

    mock_client.models.embed_content.side_effect = [
        make_mock_response([vecs[0], vecs[1]]),
        make_mock_response([vecs[2], vecs[3]]),
        make_mock_response([vecs[4]]),
    ]

    service = EmbeddingService(
        api_key="test-key",
        client=mock_client,
        batch_size=2,
        output_dimensionality=dim,
    )
    results = service.embed_texts(texts)

    assert len(results) == 5
    assert mock_client.models.embed_content.call_count == 3
    assert results == vecs


@pytest.mark.parametrize("empty_list", [[], None, "not-a-list"])
def test_embed_texts_rejects_empty_or_non_list(empty_list):
    mock_client = MagicMock()
    service = EmbeddingService(api_key="test-key", client=mock_client)

    with pytest.raises(EmbeddingConfigurationError, match="texts"):
        service.embed_texts(empty_list)


@pytest.mark.parametrize(
    "invalid_list",
    [
        ["Valid text", ""],
        ["Valid text", "   "],
        ["Valid text", None],
        ["Valid text", 12345],
    ],
)
def test_embed_texts_rejects_blank_or_invalid_item(invalid_list):
    mock_client = MagicMock()
    service = EmbeddingService(api_key="test-key", client=mock_client)

    with pytest.raises(EmbeddingConfigurationError, match="Text at index 1 must not be empty"):
        service.embed_texts(invalid_list)


# =====================================================================
# 7. Invalid Provider Responses & Malformed Vectors
# =====================================================================

def test_provider_returns_none_response():
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = None

    service = EmbeddingService(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingValidationError, match="returned None response"):
        service.embed_text("Valid text")


def test_provider_returns_no_embeddings_attribute():
    mock_client = MagicMock()
    mock_resp = MagicMock(spec=[])  # no .embeddings attribute
    mock_client.models.embed_content.return_value = mock_resp

    service = EmbeddingService(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingValidationError, match="does not contain an 'embeddings' list"):
        service.embed_text("Valid text")


def test_provider_returns_empty_embeddings_list():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.embeddings = []
    mock_client.models.embed_content.return_value = mock_resp

    service = EmbeddingService(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingValidationError, match="empty embeddings list"):
        service.embed_text("Valid text")


def test_provider_returns_fewer_embeddings_than_texts():
    mock_client = MagicMock()
    # Sent 2 texts, provider only returns 1 embedding
    mock_client.models.embed_content.return_value = make_mock_response([make_vector(768)])

    service = EmbeddingService(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingValidationError, match="Embedding count mismatch: sent 2 text\\(s\\), received 1"):
        service.embed_texts(["Text 1", "Text 2"])


def test_provider_embedding_missing_values_vector():
    mock_client = MagicMock()
    mock_emb = MagicMock(spec=[])  # no .values attribute
    mock_resp = MagicMock()
    mock_resp.embeddings = [mock_emb]
    mock_client.models.embed_content.return_value = mock_resp

    service = EmbeddingService(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingValidationError, match="missing 'values' vector"):
        service.embed_text("Valid text")


# =====================================================================
# 8. API Error Sanitization & Secret Leakage Prevention
# =====================================================================

def test_api_error_handling_sanitizes_secret_key_and_preserves_category():
    mock_client = MagicMock()
    secret_key = "AIzaSySecretApiKey123456789"
    api_error = errors.APIError(
        code=429,
        response_json={
            "message": f"Resource has been exhausted for key {secret_key}?key={secret_key}&other=123"
        },
    )
    mock_client.models.embed_content.side_effect = api_error

    service = EmbeddingService(api_key=secret_key, client=mock_client)

    with pytest.raises(EmbeddingAPIError) as exc_info:
        service.embed_text("Test query")

    err_str = str(exc_info.value)
    # Check that HTTP code and descriptive category are preserved
    assert "HTTP 429" in err_str
    assert "Rate Limit Exceeded" in err_str
    # Crucial security checks: secret key must be completely redacted
    assert secret_key not in err_str
    assert "[REDACTED_KEY]" in err_str
    assert exc_info.value.__cause__ is api_error


def test_api_error_handling_401_authentication():
    mock_client = MagicMock()
    api_error = errors.APIError(
        code=401,
        response_json={"message": "API key not valid. Please pass a valid API key."},
    )
    mock_client.models.embed_content.side_effect = api_error

    service = EmbeddingService(api_key="key", client=mock_client)
    with pytest.raises(EmbeddingAPIError) as exc_info:
        service.embed_text("Test query")

    err_str = str(exc_info.value)
    assert "HTTP 401" in err_str
    assert "Authentication Error" in err_str


def test_api_error_handling_truncates_sensitive_request_dump():
    mock_client = MagicMock()
    # Simulate a raw error dump containing sensitive patient health consultation text
    patient_text = "PATIENT_CONSULTATION_TRANSCRIPT_MRN_88776655_PRESCRIBE_HIGH_DOSE_INSULIN"
    long_raw_dump = f"Failed request payload: {patient_text} " * 10
    api_error = errors.APIError(
        code=400,
        response_json={"message": long_raw_dump},
    )
    mock_client.models.embed_content.side_effect = api_error

    service = EmbeddingService(api_key="key", client=mock_client)
    with pytest.raises(EmbeddingAPIError) as exc_info:
        service.embed_text("Test query")

    err_str = str(exc_info.value)
    assert "HTTP 400" in err_str
    assert "Invalid Request" in err_str
    assert "[truncated]" in err_str
    assert len(err_str) < 300


def test_unexpected_exception_sanitized():
    mock_client = MagicMock()
    mock_client.models.embed_content.side_effect = ConnectionResetError("https://api.google.com?key=SECRET_LEAK")

    service = EmbeddingService(api_key="test-key", client=mock_client)
    with pytest.raises(EmbeddingAPIError) as exc_info:
        service.embed_text("Test query")

    err_str = str(exc_info.value)
    assert "ConnectionResetError" in err_str
    assert "SECRET_LEAK" not in err_str
    assert isinstance(exc_info.value.__cause__, ConnectionResetError)


# =====================================================================
# 9. DocumentChunk Integration Tests
# =====================================================================

def test_embed_chunks_success():
    chunk1 = DocumentChunk(
        passage_id="DOC-01_chunk_0000_abc123",
        document_id="DOC-01",
        text="Metformin initial monotherapy for adults with T2DM.",
        source_title="Diabetes Guidelines",
        source_type="clinical_guideline",
        section="Section 1: Initial Monotherapy",
        chunk_index=0,
        metadata={"evidence_level": "Grade A"},
    )
    chunk2 = DocumentChunk(
        passage_id="DOC-01_chunk_0001_def456",
        document_id="DOC-01",
        text="Dual therapy with SGLT2 inhibitors or DPP-4 inhibitors.",
        source_title="Diabetes Guidelines",
        source_type="clinical_guideline",
        section="Section 2: Escalation",
        chunk_index=1,
        metadata={"evidence_level": "Grade B"},
    )

    dim = 256
    vec1 = make_vector(dim, start_val=0.1)
    vec2 = make_vector(dim, start_val=0.5)

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = make_mock_response([vec1, vec2])

    service = EmbeddingService(
        api_key="test-key",
        client=mock_client,
        output_dimensionality=dim,
    )
    embedded_chunks = service.embed_chunks([chunk1, chunk2])

    assert len(embedded_chunks) == 2

    # Check Chunk 1
    ec1 = embedded_chunks[0]
    assert isinstance(ec1, ChunkEmbedding)
    assert ec1.passage_id == "DOC-01_chunk_0000_abc123"
    assert ec1.embedding == vec1
    assert ec1.chunk == chunk1
    assert ec1.text == chunk1.text
    assert ec1.metadata["evidence_level"] == "Grade A"
    assert ec1.metadata["embedding_model"] == DEFAULT_EMBEDDING_MODEL
    assert ec1.metadata["embedding_dim"] == dim

    # Check Chunk 2
    ec2 = embedded_chunks[1]
    assert isinstance(ec2, ChunkEmbedding)
    assert ec2.passage_id == "DOC-01_chunk_0001_def456"
    assert ec2.embedding == vec2
    assert ec2.chunk == chunk2
    assert ec2.text == chunk2.text
    assert ec2.metadata["evidence_level"] == "Grade B"

    # Verify task_type and output_dimensionality were configured
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call_kwargs["config"].output_dimensionality == dim


@pytest.mark.parametrize("empty_chunks", [[], None, "not-a-list"])
def test_embed_chunks_rejects_empty_or_non_list(empty_chunks):
    mock_client = MagicMock()
    service = EmbeddingService(api_key="test-key", client=mock_client)

    with pytest.raises(EmbeddingConfigurationError, match="chunks"):
        service.embed_chunks(empty_chunks)


def test_embed_chunks_rejects_non_chunk_item():
    mock_client = MagicMock()
    service = EmbeddingService(api_key="test-key", client=mock_client)

    with pytest.raises(EmbeddingConfigurationError, match="is not an instance of DocumentChunk"):
        service.embed_chunks(["not-a-document-chunk"])


# =====================================================================
# 10. Security & Logging Tests (No Sensitive Clinical Text or Secrets)
# =====================================================================

def test_secrets_and_clinical_text_not_logged(caplog):
    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = make_mock_response([make_vector(768)])

    secret_key = "SUPER_SECRET_GEMINI_KEY_9999"
    sensitive_clinical_text = "PATIENT_RECORD_MRN_987654321_DIAGNOSIS_ACUTE_CORONARY_SYNDROME"

    service = EmbeddingService(api_key=secret_key, client=mock_client)

    with caplog.at_level("DEBUG"):
        service.embed_text(sensitive_clinical_text)

    # Ensure neither the secret API key nor the sensitive clinical text was logged
    assert secret_key not in caplog.text
    assert sensitive_clinical_text not in caplog.text
    assert "Embedding batch of 1 text(s)" in caplog.text
