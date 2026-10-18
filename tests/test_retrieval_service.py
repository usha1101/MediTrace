"""
Unit tests for MediTrace RetrievalService.
Verifies RAG evidence retrieval connecting EmbeddingService and VectorStore.
All tests use synthetic guideline data, in-memory Qdrant, and mocked embedding calls.
No external network or Gemini API calls are made during tests.
"""

import logging
import math
from typing import Any, Dict, List
from unittest import mock

import pytest
from qdrant_client import QdrantClient

from app.schemas import DocumentChunk, ReferencePassage
from app.services.embedding_service import (
    EmbeddingAPIError,
    EmbeddingService,
    EmbeddingServiceError,
    EmbeddingValidationError,
)
from app.services.retrieval_service import (
    DEFAULT_RETRIEVAL_TOP_K,
    RetrievalConfigurationError,
    RetrievalEmbeddingError,
    RetrievalService,
    RetrievalServiceError,
    RetrievalValidationError,
    RetrievalVectorStoreError,
    relevance_score_to_cosine_similarity,
)
from app.services.vector_store import (
    VectorStore,
    VectorStoreError,
    VectorStoreSearchError,
    cosine_similarity_to_relevance_score,
)


# =====================================================================
# Fixtures & Synthetic Helpers
# =====================================================================

def make_unit_vector(dim: int = 768, primary_idx: int = 0) -> List[float]:
    """Create a deterministic unit vector with primary activation at primary_idx."""
    raw = [0.01] * dim
    raw[primary_idx % dim] = 1.0
    norm = math.sqrt(sum(x * x for x in raw))
    return [round(x / norm, 6) for x in raw]


def make_synthetic_chunk(
    passage_id: str,
    document_id: str,
    text: str,
    source_title: str,
    source_type: str = "clinical_guideline",
    section: str = "Section 1",
    chunk_index: int = 0,
    publication_date: str = "2024-01-01",
    metadata: Dict[str, Any] = None,
) -> DocumentChunk:
    """Helper to create a valid synthetic DocumentChunk."""
    return DocumentChunk(
        passage_id=passage_id,
        document_id=document_id,
        text=text,
        source_title=source_title,
        source_type=source_type,
        section=section,
        chunk_index=chunk_index,
        publication_date=publication_date,
        metadata=metadata or {"guideline_version": "v1.0"},
    )


class FakeEmbeddingService:
    """Deterministic in-memory embedding service substitute for testing."""

    def __init__(self, vector_dim: int = 768):
        self.output_dimensionality = vector_dim
        self.query_call_count = 0
        self.last_query = None

    def embed_query(self, query: str) -> List[float]:
        self.query_call_count += 1
        self.last_query = query
        # Deterministically orient vector by length or first char
        idx = len(query) % self.output_dimensionality
        return make_unit_vector(dim=self.output_dimensionality, primary_idx=idx)


@pytest.fixture
def fake_embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService(vector_dim=768)


@pytest.fixture
def memory_vector_store() -> VectorStore:
    """Isolated in-memory VectorStore pre-populated with synthetic clinical guidelines."""
    client = QdrantClient(":memory:")
    store = VectorStore(
        collection_name="test_clinical_retrieval",
        client=client,
        vector_dim=768,
    )

    # Insert 3 distinct synthetic guideline chunks
    c1 = make_synthetic_chunk(
        passage_id="NICE_NG28_001",
        document_id="NICE_NG28",
        text="Metformin remains initial pharmacotherapy for type 2 diabetes mellitus unless contraindicated.",
        source_title="NICE Guideline NG28",
        source_type="clinical_guideline",
        section="Section 1.3 - Glycemic management",
        chunk_index=1,
    )
    c2 = make_synthetic_chunk(
        passage_id="ADA_2024_005",
        document_id="ADA_2024",
        text="First-line therapy for type 2 diabetes includes metformin and comprehensive lifestyle modifications.",
        source_title="ADA Standards of Care 2024",
        source_type="clinical_guideline",
        section="Chapter 9 - Pharmacologic approaches",
        chunk_index=5,
    )
    c3 = make_synthetic_chunk(
        passage_id="LABEL_LISINOPRIL_02",
        document_id="LABEL_LISINOPRIL",
        text="Contraindicated in patients with a history of angioedema related to previous ACE inhibitor treatment.",
        source_title="FDA Lisinopril Package Insert",
        source_type="drug_label",
        section="Contraindications",
        chunk_index=2,
    )

    # Use deterministic vectors for predictable ranking
    v1 = make_unit_vector(dim=768, primary_idx=10)
    v2 = make_unit_vector(dim=768, primary_idx=10)  # aligned with v1
    v3 = make_unit_vector(dim=768, primary_idx=500) # distinct direction

    store.upsert_chunks([c1, c2, c3], [v1, v2, v3])
    return store


@pytest.fixture
def retrieval_service(
    fake_embedding_service: FakeEmbeddingService,
    memory_vector_store: VectorStore,
) -> RetrievalService:
    """RetrievalService instance wired to fake embedding and memory vector store."""
    return RetrievalService(
        embedding_service=fake_embedding_service,
        vector_store=memory_vector_store,
        default_top_k=5,
    )


# =====================================================================
# 1. Initialization & Configuration Tests
# =====================================================================

def test_init_defaults(fake_embedding_service, memory_vector_store):
    """Verify default initialization parameters."""
    service = RetrievalService(
        embedding_service=fake_embedding_service,
        vector_store=memory_vector_store,
    )
    assert service.default_top_k == DEFAULT_RETRIEVAL_TOP_K
    assert service.default_similarity_threshold is None
    assert service.embedding_service is fake_embedding_service
    assert service.vector_store is memory_vector_store


def test_init_custom_parameters(fake_embedding_service, memory_vector_store):
    """Verify initialization with custom top_k and threshold."""
    service = RetrievalService(
        embedding_service=fake_embedding_service,
        vector_store=memory_vector_store,
        default_top_k=10,
        default_similarity_threshold=0.75,
    )
    assert service.default_top_k == 10
    assert service.default_similarity_threshold == 0.75


def test_init_rejects_invalid_dependencies():
    """Verify invalid dependency objects lacking required methods are rejected."""
    # Invalid embedding service missing embed_query
    with pytest.raises(RetrievalConfigurationError, match="embedding_service must provide a callable embed_query"):
        RetrievalService(embedding_service="not_a_service", vector_store=mock.Mock(spec=VectorStore))

    # Invalid vector store missing search
    mock_emb = mock.Mock()
    mock_emb.embed_query = lambda q: [0.1] * 768
    with pytest.raises(RetrievalConfigurationError, match="vector_store must provide a callable search"):
        RetrievalService(embedding_service=mock_emb, vector_store="not_a_store")


@pytest.mark.parametrize("bad_top_k", [0, -1, -10, "5", True, False, None])
def test_init_rejects_invalid_default_top_k(fake_embedding_service, memory_vector_store, bad_top_k):
    """Verify default_top_k must be a positive integer and rejects bools and None."""
    with pytest.raises(RetrievalConfigurationError, match="default_top_k must be a positive integer"):
        RetrievalService(
            embedding_service=fake_embedding_service,
            vector_store=memory_vector_store,
            default_top_k=bad_top_k,
        )


@pytest.mark.parametrize("bad_thresh", [-0.1, 1.01, 2.0, -1.0, float("nan"), float("inf"), -float("inf"), True, False, "0.5"])
def test_init_rejects_invalid_default_similarity_threshold(
    fake_embedding_service, memory_vector_store, bad_thresh
):
    """Verify default_similarity_threshold must be between 0.0 and 1.0."""
    with pytest.raises(RetrievalConfigurationError):
        RetrievalService(
            embedding_service=fake_embedding_service,
            vector_store=memory_vector_store,
            default_similarity_threshold=bad_thresh,
        )


def test_init_rejects_dimension_mismatch(memory_vector_store):
    """Verify that dimension mismatch between embedding_service and vector_store is rejected on init."""
    mismatched_emb = FakeEmbeddingService(vector_dim=512)
    with pytest.raises(
        RetrievalConfigurationError,
        match=r"Embedding service output dimension \(512\) does not match vector store dimension \(768\)",
    ):
        RetrievalService(
            embedding_service=mismatched_emb,
            vector_store=memory_vector_store,
        )


def test_lazy_initialization_missing_api_key_raises_sanitized_error(monkeypatch):
    """Verify lazy initialization without GEMINI_API_KEY raises sanitized RetrievalConfigurationError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = RetrievalService()
    with pytest.raises(
        RetrievalConfigurationError,
        match=r"Failed to initialize default EmbeddingService: EmbeddingConfigurationError",
    ):
        _ = service.embedding_service


def test_lazy_initialization_with_credentials(monkeypatch):
    """Verify lazy initialization succeeds when environment has credentials and client is mocked."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-mock-api-key")
    with mock.patch("app.services.embedding_service.genai.Client"):
        service = RetrievalService()
        emb = service.embedding_service
        assert isinstance(emb, EmbeddingService)
        assert emb.output_dimensionality == 768


# =====================================================================
# 2. Query Input Validation Tests
# =====================================================================

@pytest.mark.parametrize("empty_query", ["", "   ", "\n\t  \r", None, 123, ["query"], True, False])
def test_retrieve_rejects_empty_or_non_string_query(retrieval_service, empty_query):
    """Verify retrieve explicitly rejects empty or non-string queries."""
    with pytest.raises(RetrievalValidationError, match="Query must be a non-empty string"):
        retrieval_service.retrieve(empty_query)


@pytest.mark.parametrize("bad_top_k", [0, -1, -5, "10", True, False, 3.5])
def test_retrieve_rejects_invalid_top_k(retrieval_service, bad_top_k):
    """Verify retrieve explicitly rejects non-positive or bool top_k."""
    with pytest.raises(RetrievalValidationError, match="top_k must be a positive integer"):
        retrieval_service.retrieve("metformin indications", top_k=bad_top_k)


@pytest.mark.parametrize("bad_thresh", [-0.01, 1.01, 2.5, -1.0, float("nan"), float("inf"), -float("inf"), True, False, "0.7"])
def test_retrieve_rejects_invalid_similarity_threshold(retrieval_service, bad_thresh):
    """Verify retrieve explicitly rejects out-of-bound or non-float similarity thresholds."""
    with pytest.raises(RetrievalValidationError):
        retrieval_service.retrieve("metformin indications", similarity_threshold=bad_thresh)


@pytest.mark.parametrize("bad_filter", ["", "   ", "\t", True, False, 123])
def test_retrieve_rejects_invalid_metadata_filters(retrieval_service, bad_filter):
    """Verify retrieve explicitly rejects empty or non-string metadata filters."""
    with pytest.raises(RetrievalValidationError, match="document_id filter must be a non-empty string"):
        retrieval_service.retrieve("diabetes", document_id=bad_filter)

    with pytest.raises(RetrievalValidationError, match="source_type filter must be a non-empty string"):
        retrieval_service.retrieve("diabetes", source_type=bad_filter)


def test_retrieve_uses_default_top_k_when_none(retrieval_service):
    """Verify top_k=None falls back to self.default_top_k."""
    retrieval_service.default_top_k = 2
    passages = retrieval_service.retrieve("diabetes", top_k=None)
    assert len(passages) <= 2


def test_sanitized_validation_error_messages(retrieval_service):
    """Verify invalid top_k or similarity_threshold values do not echo raw sensitive parameter strings."""
    sensitive_param = "PATIENT_SSN_999-99-9999_DIAGNOSIS"

    with pytest.raises(RetrievalValidationError) as exc_info:
        retrieval_service.retrieve("query", top_k=sensitive_param)
    assert sensitive_param not in str(exc_info.value)
    assert "top_k must be a positive integer (not bool), got str" in str(exc_info.value)

    with pytest.raises(RetrievalValidationError) as exc_info:
        retrieval_service.retrieve("query", similarity_threshold=sensitive_param)
    assert sensitive_param not in str(exc_info.value)
    assert "similarity_threshold must be a numeric float, got str" in str(exc_info.value)


# =====================================================================
# 3. Successful Retrieval & Ranking Preservation
# =====================================================================

def test_successful_retrieval(retrieval_service, fake_embedding_service):
    """Verify successful retrieval queries embedding service and vector store."""
    results = retrieval_service.retrieve("first line treatment type 2 diabetes", top_k=2)

    assert fake_embedding_service.query_call_count == 1
    assert len(results) <= 2
    assert all(isinstance(p, ReferencePassage) for p in results)


def test_ranking_preservation(retrieval_service):
    """
    Verify returned passages maintain descending order by relevance score.
    """
    # Make a mock embedding service that produces vector aligned with NICE and ADA guidelines (idx=10)
    mock_emb = mock.Mock(spec=EmbeddingService)
    mock_emb.embed_query.return_value = make_unit_vector(dim=768, primary_idx=10)

    service = RetrievalService(
        embedding_service=mock_emb,
        vector_store=retrieval_service.vector_store,
        default_top_k=3,
    )

    passages = service.retrieve("diabetes first line", top_k=3)
    assert len(passages) == 3

    # Scores must be in descending order
    scores = [p.relevance_score for p in passages]
    assert scores == sorted(scores, reverse=True)
    # The first two passages (aligned on idx=10) should have high scores (~1.0)
    assert scores[0] >= 0.99
    assert scores[1] >= 0.99
    # The lisinopril passage (idx=500) has significantly lower score
    assert scores[2] < scores[0]


def test_search_alias_matches_retrieve(retrieval_service):
    """Verify search() method behaves identically to retrieve()."""
    r1 = retrieval_service.retrieve("diabetes management", top_k=2)
    r2 = retrieval_service.search("diabetes management", top_k=2)
    assert [p.passage_id for p in r1] == [p.passage_id for p in r2]


# =====================================================================
# 4. Metadata Preservation & Provenance
# =====================================================================

def test_metadata_and_provenance_preservation(retrieval_service):
    """Verify all document metadata and provenance fields are preserved."""
    # Aligns with lisinopril passage (idx=500)
    mock_emb = mock.Mock(spec=EmbeddingService)
    mock_emb.embed_query.return_value = make_unit_vector(dim=768, primary_idx=500)

    service = RetrievalService(
        embedding_service=mock_emb,
        vector_store=retrieval_service.vector_store,
        default_top_k=1,
    )

    passages = service.retrieve("angioedema contraindications", top_k=1)
    assert len(passages) == 1
    passage = passages[0]

    assert passage.passage_id == "LABEL_LISINOPRIL_02"
    assert passage.source_document == "FDA Lisinopril Package Insert"
    assert passage.section_or_page == "Contraindications"
    assert "angioedema" in passage.content.lower()
    assert passage.title == "FDA Lisinopril Package Insert - Contraindications"
    assert passage.relevance_score is not None
    assert 0.0 <= passage.relevance_score <= 1.0

    meta = passage.retrieval_metadata
    assert meta["document_id"] == "LABEL_LISINOPRIL"
    assert meta["source_type"] == "drug_label"
    assert meta["chunk_index"] == 2


# =====================================================================
# 5. Filtering (document_id and source_type)
# =====================================================================

def test_filtering_by_document_id(retrieval_service):
    """Verify filtering by document_id returns only passages from that document and excludes others."""
    passages = retrieval_service.retrieve(
        "diabetes treatment",
        document_id="NICE_NG28",
        top_k=5,
    )
    assert len(passages) == 1
    assert passages[0].passage_id == "NICE_NG28_001"
    assert passages[0].retrieval_metadata["document_id"] == "NICE_NG28"


def test_filtering_by_source_type(retrieval_service):
    """Verify filtering by source_type returns only passages matching that type."""
    passages = retrieval_service.retrieve(
        "guideline or label",
        source_type="drug_label",
        top_k=5,
    )
    assert len(passages) == 1
    assert passages[0].retrieval_metadata["source_type"] == "drug_label"
    assert passages[0].passage_id == "LABEL_LISINOPRIL_02"


def test_retrieve_guidelines_convenience_method(retrieval_service):
    """Verify retrieve_guidelines applies source_type='clinical_guideline' filter and excludes drug labels."""
    passages = retrieval_service.retrieve_guidelines("diabetes therapy", top_k=5)
    assert len(passages) == 2
    assert all(p.retrieval_metadata["source_type"] == "clinical_guideline" for p in passages)
    passage_ids = [p.passage_id for p in passages]
    assert "NICE_NG28_001" in passage_ids
    assert "ADA_2024_005" in passage_ids
    assert "LABEL_LISINOPRIL_02" not in passage_ids


def test_filtering_by_both_document_id_and_source_type(retrieval_service):
    """Verify combining document_id and source_type filters returns only matching passages."""
    passages = retrieval_service.retrieve(
        "guideline query",
        document_id="NICE_NG28",
        source_type="clinical_guideline",
        top_k=5,
    )
    assert len(passages) == 1
    assert passages[0].passage_id == "NICE_NG28_001"
    assert passages[0].retrieval_metadata["document_id"] == "NICE_NG28"
    assert passages[0].retrieval_metadata["source_type"] == "clinical_guideline"

    # Conflicting filter matches nothing
    mismatched = retrieval_service.retrieve(
        "guideline query",
        document_id="NICE_NG28",
        source_type="drug_label",
        top_k=5,
    )
    assert mismatched == []


def test_filter_matching_nothing_returns_empty_list(retrieval_service):
    """Verify filters matching no indexed documents return empty list."""
    passages = retrieval_service.retrieve(
        "diabetes",
        document_id="NON_EXISTENT_GUIDELINE",
    )
    assert passages == []


# =====================================================================
# 6. Similarity Threshold Filtering
# =====================================================================

def test_similarity_threshold_filtering(retrieval_service):
    """Verify similarity_threshold prunes passages below the threshold."""
    mock_emb = mock.Mock(spec=EmbeddingService)
    # Query vector matches idx=10 (NICE and ADA scores ~1.0; Lisinopril idx=500 score ~0.5)
    mock_emb.embed_query.return_value = make_unit_vector(dim=768, primary_idx=10)

    service = RetrievalService(
        embedding_service=mock_emb,
        vector_store=retrieval_service.vector_store,
        default_top_k=5,
    )

    # 1. No threshold: returns all 3 passages
    all_passages = service.retrieve("diabetes", similarity_threshold=None)
    assert len(all_passages) == 3

    # 2. Strict threshold of 0.8: returns only the 2 high-scoring diabetes passages
    filtered_passages = service.retrieve("diabetes", similarity_threshold=0.8)
    assert len(filtered_passages) == 2
    assert all(p.relevance_score >= 0.8 for p in filtered_passages)

    # 3. Query with low similarity to all passages: threshold of 0.8 returns empty list
    mock_emb.embed_query.return_value = make_unit_vector(dim=768, primary_idx=99)
    unrelated_passages = service.retrieve("unrelated query", similarity_threshold=0.8)
    assert unrelated_passages == []


def test_default_similarity_threshold_applied(fake_embedding_service, memory_vector_store):
    """Verify default_similarity_threshold set at init is applied when not overridden."""
    service = RetrievalService(
        embedding_service=fake_embedding_service,
        vector_store=memory_vector_store,
        default_top_k=5,
        default_similarity_threshold=0.999,  # Very high default threshold
    )
    passages = service.retrieve("diabetes treatment")
    assert passages == []


@pytest.mark.parametrize(
    "raw_cosine,expected_relevance",
    [
        (1.0, 1.0),
        (0.0, 0.5),
        (-0.2, 0.4),
        (-0.8, 0.1),
        (-1.0, 0.0),
    ],
)
def test_similarity_score_roundtrip_consistency(raw_cosine, expected_relevance):
    """Verify bidirectional mathematical consistency between cosine and relevance conversions."""
    rel = cosine_similarity_to_relevance_score(raw_cosine)
    assert rel == pytest.approx(expected_relevance, abs=1e-4)

    raw_back = relevance_score_to_cosine_similarity(rel)
    assert raw_back == pytest.approx(raw_cosine, abs=1e-4)


def test_relevance_score_to_cosine_similarity_edge_cases():
    """Verify edge cases for relevance_score_to_cosine_similarity."""
    assert relevance_score_to_cosine_similarity(None) is None
    assert relevance_score_to_cosine_similarity(float("nan")) is None
    assert relevance_score_to_cosine_similarity(float("inf")) is None
    assert relevance_score_to_cosine_similarity(-float("inf")) is None
    assert relevance_score_to_cosine_similarity("invalid") is None
    # Out of range clamped safely to [-1.0, 1.0]
    assert relevance_score_to_cosine_similarity(-0.5) == -1.0
    assert relevance_score_to_cosine_similarity(1.5) == 1.0


def test_similarity_threshold_boundary_zero_and_one(retrieval_service):
    """Verify similarity_threshold=0.0 includes all items and similarity_threshold=1.0 only includes exact matches."""
    # 0.0 threshold: includes all 3 passages
    passages_zero = retrieval_service.retrieve("diabetes", similarity_threshold=0.0)
    assert len(passages_zero) == 3
    assert all(p.relevance_score is not None and p.relevance_score >= 0.0 for p in passages_zero)

    # 1.0 threshold: with query aligned with NICE and ADA (scores ~1.0)
    mock_emb = mock.Mock(spec=EmbeddingService)
    mock_emb.embed_query.return_value = make_unit_vector(dim=768, primary_idx=10)
    service = RetrievalService(
        embedding_service=mock_emb,
        vector_store=retrieval_service.vector_store,
    )
    passages_one = service.retrieve("diabetes", similarity_threshold=1.0)
    # Only items whose normalized score rounds to 1.0 are returned
    assert len(passages_one) == 2
    assert all(p.relevance_score == 1.0 for p in passages_one)
    assert "LABEL_LISINOPRIL_02" not in [p.passage_id for p in passages_one]


def test_similarity_threshold_exact_boundary_match(retrieval_service):
    """Verify a passage with relevance_score exactly equal to the threshold is retained."""
    mock_emb = mock.Mock(spec=EmbeddingService)
    mock_emb.embed_query.return_value = make_unit_vector(dim=768, primary_idx=10)
    service = RetrievalService(
        embedding_service=mock_emb,
        vector_store=retrieval_service.vector_store,
    )
    # Retrieve without threshold to inspect exact scores
    unfiltered = service.retrieve("diabetes")
    lowest_score = unfiltered[-1].relevance_score
    assert lowest_score is not None

    # Filter with threshold set exactly to lowest_score
    filtered = service.retrieve("diabetes", similarity_threshold=lowest_score)
    # The item with score == lowest_score MUST be included (boundary inclusive >=)
    assert any(p.relevance_score == lowest_score for p in filtered)
    assert len(filtered) == 3


# =====================================================================
# 7. Error Handling & Sanitized Logging
# =====================================================================

def test_retrieve_rejects_runtime_dimension_mismatch(memory_vector_store):
    """Verify that if query vector dimensionality does not match vector store, an error is raised."""
    mock_emb = mock.Mock()
    # Returns 512 dimensions instead of 768
    mock_emb.embed_query = lambda q: [0.1] * 512
    service = RetrievalService(
        embedding_service=mock_emb,
        vector_store=memory_vector_store,
    )
    with pytest.raises(RetrievalConfigurationError, match="Generated query vector dimension"):
        service.retrieve("test query")


def test_embedding_service_failure_raises_retrieval_embedding_error(retrieval_service):
    """Verify EmbeddingService failures are caught and wrapped in RetrievalEmbeddingError."""
    with mock.patch.object(
        retrieval_service.embedding_service,
        "embed_query",
        side_effect=EmbeddingAPIError("API 503 Service Unavailable"),
    ):
        with pytest.raises(RetrievalEmbeddingError, match="Failed to generate query embedding: EmbeddingAPIError"):
            retrieval_service.retrieve("clinical query")


def test_vector_store_failure_raises_retrieval_vector_store_error(retrieval_service):
    """Verify VectorStore failures are caught and wrapped in RetrievalVectorStoreError."""
    with mock.patch.object(
        retrieval_service.vector_store,
        "search",
        side_effect=VectorStoreSearchError("Storage connection closed"),
    ):
        with pytest.raises(RetrievalVectorStoreError, match="Failed to search vector store: VectorStoreSearchError"):
            retrieval_service.retrieve("clinical query")


def test_sensitive_query_text_never_logged(caplog, retrieval_service):
    """
    Verify that raw clinical queries, patient identifiers, or sensitive terms
    are NEVER logged by RetrievalService under any log level.
    """
    sensitive_query = "PATIENT_DOB_1980_DIAGNOSED_WITH_STAGE_4_RENAL_FAILURE_SECRET"

    with caplog.at_level(logging.DEBUG):
        results = retrieval_service.retrieve(sensitive_query)
        assert isinstance(results, list)

    # The sensitive string must never appear in log records
    assert sensitive_query not in caplog.text


def test_sensitive_content_never_logged_on_error(caplog, retrieval_service):
    """
    Verify that sensitive query text is not logged when an error occurs during retrieval.
    """
    sensitive_query = "PATIENT_SECRET_ALLERGY_ANAPHYLAXIS_DO_NOT_LOG"

    with mock.patch.object(
        retrieval_service.embedding_service,
        "embed_query",
        side_effect=EmbeddingAPIError("Network timeout"),
    ):
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(RetrievalEmbeddingError):
                retrieval_service.retrieve(sensitive_query)

    assert sensitive_query not in caplog.text
    # Only safe exception type is logged
    assert "EmbeddingAPIError" in caplog.text


def test_empty_results_when_vector_store_is_empty():
    """Verify empty vector store returns empty list safely."""
    empty_store = VectorStore(
        collection_name="empty_test_col",
        client=QdrantClient(":memory:"),
        vector_dim=768,
    )
    fake_emb = FakeEmbeddingService(vector_dim=768)
    service = RetrievalService(embedding_service=fake_emb, vector_store=empty_store)

    results = service.retrieve("any query")
    assert results == []
