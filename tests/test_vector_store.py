"""
Unit tests for MediTrace VectorStore service.
All tests use in-memory QdrantClient (':memory:') and synthetic guideline data.
No external Qdrant server or Gemini network calls are made.
"""

import logging
import math
from typing import Any, Dict, List
import uuid

import pytest
from qdrant_client import QdrantClient, models

from unittest import mock

from app.schemas import DocumentChunk, ReferencePassage
from app.services.embedding_service import ChunkEmbedding
from app.services.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_VECTOR_DIMENSION,
    VectorStore,
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreInsertionError,
    VectorStoreSearchError,
    VectorStoreValidationError,
    cosine_similarity_to_relevance_score,
    passage_id_to_point_id,
)


# =====================================================================
# Test Helpers & Fixtures
# =====================================================================

def make_test_vector(dim: int = DEFAULT_VECTOR_DIMENSION, primary_index: int = 0) -> List[float]:
    """
    Generate a deterministic normalized vector with primary activation at primary_index.
    """
    raw = [0.01] * dim
    raw[primary_index % dim] = 1.0
    norm = math.sqrt(sum(x * x for x in raw))
    return [round(x / norm, 6) for x in raw]


def make_test_chunk(
    passage_id: str = "NICE_NG28_chunk_0000",
    document_id: str = "DOC_NICE_NG28",
    text: str = "Metformin is recommended as the first-line pharmacotherapy for type 2 diabetes mellitus.",
    source_title: str = "NICE Guideline NG28",
    source_type: str = "clinical_guideline",
    section: str = "Section 1.3 - Pharmacological management",
    chunk_index: int = 0,
    publication_date: str = "2024-01-15",
    metadata: Dict[str, Any] = None,
) -> DocumentChunk:
    """Create a synthetic DocumentChunk instance for testing."""
    return DocumentChunk(
        passage_id=passage_id,
        document_id=document_id,
        text=text,
        source_title=source_title,
        source_type=source_type,
        section=section,
        chunk_index=chunk_index,
        publication_date=publication_date,
        metadata=metadata or {"guideline_topic": "Type 2 Diabetes"},
    )


@pytest.fixture
def memory_client() -> QdrantClient:
    """Fixture providing an isolated in-memory Qdrant client."""
    return QdrantClient(":memory:")


@pytest.fixture
def memory_store(memory_client: QdrantClient) -> VectorStore:
    """Fixture providing a fresh in-memory VectorStore."""
    return VectorStore(
        collection_name="test_guidelines",
        client=memory_client,
        vector_dim=DEFAULT_VECTOR_DIMENSION,
    )


# =====================================================================
# 1. Initialization & Configuration Tests
# =====================================================================

def test_init_default_configuration():
    """Verify VectorStore initializes with default collection name and dimension."""
    client = QdrantClient(":memory:")
    store = VectorStore(client=client)
    assert store.collection_name == DEFAULT_COLLECTION_NAME
    assert store.vector_dim == DEFAULT_VECTOR_DIMENSION
    assert client.collection_exists(DEFAULT_COLLECTION_NAME) is True

    info = client.get_collection(DEFAULT_COLLECTION_NAME)
    assert info.config.params.vectors.size == DEFAULT_VECTOR_DIMENSION
    assert info.config.params.vectors.distance == models.Distance.COSINE


def test_init_with_memory_path():
    """Verify VectorStore initializes cleanly with storage_path=':memory:'."""
    store = VectorStore(storage_path=":memory:", collection_name="memory_test")
    assert store.client.collection_exists("memory_test") is True
    store.close()


def test_init_rejects_empty_or_whitespace_collection_name():
    """Verify collection name must be a non-empty string."""
    with pytest.raises(VectorStoreConfigurationError, match="Collection name must be a non-empty string"):
        VectorStore(client=QdrantClient(":memory:"), collection_name="")

    with pytest.raises(VectorStoreConfigurationError, match="Collection name must be a non-empty string"):
        VectorStore(client=QdrantClient(":memory:"), collection_name="   ")

    with pytest.raises(VectorStoreConfigurationError, match="Collection name must be a non-empty string"):
        VectorStore(client=QdrantClient(":memory:"), collection_name=None)


def test_init_rejects_invalid_vector_dimension():
    """Verify vector dimension must be a positive integer and rejects bools."""
    with pytest.raises(VectorStoreConfigurationError, match="Vector dimension must be a positive integer"):
        VectorStore(client=QdrantClient(":memory:"), vector_dim=0)

    with pytest.raises(VectorStoreConfigurationError, match="Vector dimension must be a positive integer"):
        VectorStore(client=QdrantClient(":memory:"), vector_dim=-10)

    with pytest.raises(VectorStoreConfigurationError, match="Vector dimension must be a positive integer"):
        VectorStore(client=QdrantClient(":memory:"), vector_dim="768")

    # Reject bool explicitly even though bool is a subclass of int
    with pytest.raises(VectorStoreConfigurationError, match="Vector dimension must be a positive integer"):
        VectorStore(client=QdrantClient(":memory:"), vector_dim=True)

    with pytest.raises(VectorStoreConfigurationError, match="Vector dimension must be a positive integer"):
        VectorStore(client=QdrantClient(":memory:"), vector_dim=False)


def test_existing_collection_dimension_mismatch_raises_error():
    """Verify VectorStore detects and rejects existing collections with conflicting dimension."""
    client = QdrantClient(":memory:")
    # Pre-create collection with 512 dimensions
    client.create_collection(
        collection_name="conflict_col",
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
    )

    with pytest.raises(VectorStoreConfigurationError, match="exists with vector dimension 512"):
        VectorStore(
            collection_name="conflict_col",
            client=client,
            vector_dim=768,
        )


def test_existing_collection_distance_mismatch_raises_error():
    """Verify VectorStore detects and rejects existing collections with non-cosine distance metric."""
    client = QdrantClient(":memory:")
    # Pre-create collection with 768 dimensions but Euclid distance metric
    client.create_collection(
        collection_name="euclid_col",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.EUCLID),
    )

    with pytest.raises(VectorStoreConfigurationError, match="exists with distance metric"):
        VectorStore(
            collection_name="euclid_col",
            client=client,
            vector_dim=768,
        )


def test_existing_collection_valid_configuration_reused_cleanly():
    """Verify existing collections with matching dimension and cosine distance are reused without error."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="matching_col",
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )

    store = VectorStore(
        collection_name="matching_col",
        client=client,
        vector_dim=768,
    )
    assert store.vector_dim == 768
    assert client.collection_exists("matching_col") is True


def test_ensure_collection_force_recreate():
    """Verify force_recreate drops existing collection and establishes correct schema."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="recreate_col",
        vectors_config=models.VectorParams(size=256, distance=models.Distance.DOT),
    )

    store = VectorStore(
        collection_name="recreate_col",
        client=client,
        vector_dim=768,
        auto_ensure_collection=False,
    )
    store.ensure_collection(force_recreate=True)

    info = client.get_collection("recreate_col")
    assert info.config.params.vectors.size == 768
    assert info.config.params.vectors.distance == models.Distance.COSINE


# =====================================================================
# 2. Point ID & Helper Function Tests
# =====================================================================

def test_passage_id_to_point_id_deterministic_uuid():
    """Verify passage_id_to_point_id generates stable, valid UUID strings."""
    pid = "DOC-01_sec-1_chunk_00"
    uid1 = passage_id_to_point_id(pid)
    uid2 = passage_id_to_point_id(pid)
    assert uid1 == uid2
    # Verify it is valid UUID format
    assert str(uuid.UUID(uid1)) == uid1


def test_passage_id_to_point_id_preserves_existing_uuid():
    """Verify already-valid UUID strings are preserved."""
    raw_uuid = str(uuid.uuid4())
    converted = passage_id_to_point_id(raw_uuid)
    assert converted == raw_uuid


def test_passage_id_to_point_id_handles_integer():
    """Verify non-negative integers are formatted correctly."""
    assert passage_id_to_point_id(12345) == "12345"
    with pytest.raises(VectorStoreValidationError, match="must be non-negative"):
        passage_id_to_point_id(-5)


def test_passage_id_to_point_id_rejects_empty():
    """Verify empty or whitespace string IDs are rejected."""
    with pytest.raises(VectorStoreValidationError, match="Passage ID must be a non-empty string"):
        passage_id_to_point_id("")
    with pytest.raises(VectorStoreValidationError, match="Passage ID must be a non-empty string"):
        passage_id_to_point_id("   ")


def test_passage_id_to_point_id_rejects_bool():
    """Verify boolean values are rejected for passage IDs."""
    with pytest.raises(VectorStoreValidationError, match="Passage ID must not be a boolean value"):
        passage_id_to_point_id(True)
    with pytest.raises(VectorStoreValidationError, match="Passage ID must not be a boolean value"):
        passage_id_to_point_id(False)


# =====================================================================
# 3. Vector Validation & Rejection Tests
# =====================================================================

def test_validate_vector_dimension_mismatch(memory_store: VectorStore):
    """Verify vector with incorrect length is rejected."""
    wrong_dim_vector = [0.1] * 512
    with pytest.raises(VectorStoreValidationError, match="Dimension mismatch for query_vector: expected 768, got 512"):
        memory_store.search(wrong_dim_vector)


@pytest.mark.parametrize(
    "bad_vector, err_msg",
    [
        (None, "Expected query_vector to be a list of floats"),
        ("not-a-vector", "Expected query_vector to be a list of floats"),
        ([0.1] * 767 + ["not-a-float"], "is not a valid numeric float"),
        ([0.1] * 767 + [True], "is not a valid numeric float"),
        ([0.1] * 767 + [float("nan")], "is non-finite"),
        ([0.1] * 767 + [float("inf")], "is non-finite"),
        ([0.1] * 767 + [-float("inf")], "is non-finite"),
    ],
)
def test_validate_vector_rejects_malformed_values(memory_store: VectorStore, bad_vector, err_msg):
    """Verify non-numeric or non-finite elements in vectors are rejected."""
    with pytest.raises(VectorStoreValidationError, match=err_msg):
        memory_store.search(bad_vector)


# =====================================================================
# 4. Vector Insertion (upsert_embeddings & upsert_chunks)
# =====================================================================

def test_upsert_embeddings_successful(memory_store: VectorStore):
    """Verify upsert_embeddings successfully stores points with full payload."""
    chunk = make_test_chunk(passage_id="chunk_1")
    vector = make_test_vector(primary_index=0)
    embedding = ChunkEmbedding(
        passage_id="chunk_1",
        embedding=vector,
        chunk=chunk,
        metadata={"embed_source": "test_suite"},
    )

    count = memory_store.upsert_embeddings([embedding])
    assert count == 1
    assert memory_store.count() == 1

    # Verify retrieval
    retrieved = memory_store.get_by_id("chunk_1")
    assert retrieved is not None
    assert retrieved.passage_id == "chunk_1"
    assert retrieved.content == chunk.text
    assert retrieved.source_document == chunk.source_title
    assert retrieved.section_or_page == chunk.section
    assert retrieved.retrieval_metadata["document_id"] == chunk.document_id
    assert retrieved.retrieval_metadata["source_type"] == chunk.source_type
    assert retrieved.retrieval_metadata["publication_date"] == chunk.publication_date
    assert retrieved.retrieval_metadata["embed_source"] == "test_suite"


def test_upsert_chunks_convenience_method(memory_store: VectorStore):
    """Verify upsert_chunks works identically when given parallel chunks and vectors."""
    c1 = make_test_chunk(passage_id="c1", document_id="doc_a")
    c2 = make_test_chunk(passage_id="c2", document_id="doc_b")
    v1 = make_test_vector(primary_index=1)
    v2 = make_test_vector(primary_index=2)

    count = memory_store.upsert_chunks([c1, c2], [v1, v2])
    assert count == 2
    assert memory_store.count() == 2
    assert memory_store.count(document_id="doc_a") == 1
    assert memory_store.count(document_id="doc_b") == 1


def test_upsert_idempotence_and_overwriting(memory_store: VectorStore):
    """Verify re-upserting same passage_id updates rather than creating duplicate points."""
    chunk_v1 = make_test_chunk(passage_id="p_update", text="Version 1 guideline text.")
    chunk_v2 = make_test_chunk(passage_id="p_update", text="Version 2 updated guideline text.")
    v1 = make_test_vector(primary_index=5)
    v2 = make_test_vector(primary_index=6)

    memory_store.upsert_chunks([chunk_v1], [v1])
    assert memory_store.count() == 1
    r1 = memory_store.get_by_id("p_update")
    assert r1.content == "Version 1 guideline text."

    # Update with v2
    memory_store.upsert_chunks([chunk_v2], [v2])
    assert memory_store.count() == 1  # count remains 1, no duplicate
    r2 = memory_store.get_by_id("p_update")
    assert r2.content == "Version 2 updated guideline text."


def test_upsert_rejects_empty_and_invalid_inputs(memory_store: VectorStore):
    """Verify empty lists, non-lists, and mismatched inputs are rejected."""
    with pytest.raises(VectorStoreValidationError, match="must be a non-empty list of ChunkEmbedding"):
        memory_store.upsert_embeddings([])

    with pytest.raises(VectorStoreValidationError, match="must be a non-empty list of ChunkEmbedding"):
        memory_store.upsert_embeddings(None)

    with pytest.raises(VectorStoreValidationError, match="must be an instance of ChunkEmbedding"):
        memory_store.upsert_embeddings(["not_an_embedding"])

    # upsert_chunks length mismatch
    c = make_test_chunk()
    v1 = make_test_vector(primary_index=0)
    v2 = make_test_vector(primary_index=1)
    with pytest.raises(VectorStoreValidationError, match="Length mismatch: 1 chunks vs 2 embeddings"):
        memory_store.upsert_chunks([c], [v1, v2])


# =====================================================================
# 5. Search Ranking & Limits
# =====================================================================

def test_similarity_search_ranking(memory_store: VectorStore):
    """
    Verify search ranking orders results by cosine similarity to query vector.
    """
    # Point A has primary activation at 0
    # Point B has primary activation at 50
    # Point C has primary activation at 100
    chunk_a = make_test_chunk(passage_id="target_match", text="Metformin target passage.")
    chunk_b = make_test_chunk(passage_id="distant_match", text="Insulin secondary passage.")
    chunk_c = make_test_chunk(passage_id="unrelated_match", text="Asthma inhaler passage.")

    va = make_test_vector(primary_index=0)
    vb = make_test_vector(primary_index=50)
    vc = make_test_vector(primary_index=100)

    memory_store.upsert_chunks([chunk_a, chunk_b, chunk_c], [va, vb, vc])

    # Query closely aligns with Point A
    query_vector = make_test_vector(primary_index=0)
    results = memory_store.search(query_vector=query_vector, top_k=3)

    assert len(results) == 3
    # First result must be target_match with score close to 1.0
    assert results[0].passage_id == "target_match"
    assert results[0].relevance_score >= 0.99
    assert results[0].relevance_score >= results[1].relevance_score
    assert results[1].relevance_score >= results[2].relevance_score


def test_search_top_k_limit(memory_store: VectorStore):
    """Verify top_k strictly limits the number of returned passages."""
    chunks = [make_test_chunk(passage_id=f"p_{i}") for i in range(10)]
    vectors = [make_test_vector(primary_index=i) for i in range(10)]
    memory_store.upsert_chunks(chunks, vectors)

    query = make_test_vector(primary_index=0)

    res_1 = memory_store.search(query, top_k=1)
    assert len(res_1) == 1

    res_4 = memory_store.search(query, top_k=4)
    assert len(res_4) == 4

    res_10 = memory_store.search(query, top_k=20)
    assert len(res_10) == 10  # Only 10 available


def test_search_rejects_invalid_top_k(memory_store: VectorStore):
    """Verify top_k <= 0 or non-int is rejected."""
    query = make_test_vector(primary_index=0)
    with pytest.raises(VectorStoreValidationError, match="top_k must be a positive integer"):
        memory_store.search(query, top_k=0)

    with pytest.raises(VectorStoreValidationError, match="top_k must be a positive integer"):
        memory_store.search(query, top_k=-2)

    with pytest.raises(VectorStoreValidationError, match="top_k must be a positive integer"):
        memory_store.search(query, top_k="5")

    # Reject bool explicitly even though bool is a subclass of int
    with pytest.raises(VectorStoreValidationError, match="top_k must be a positive integer"):
        memory_store.search(query, top_k=True)

    with pytest.raises(VectorStoreValidationError, match="top_k must be a positive integer"):
        memory_store.search(query, top_k=False)


def test_cosine_similarity_normalization_helper():
    """
    Verify cosine_similarity_to_relevance_score maps [-1.0, 1.0] to [0.0, 1.0]
    monotonically without squashing negative scores to zero.
    """
    assert cosine_similarity_to_relevance_score(1.0) == 1.0
    assert cosine_similarity_to_relevance_score(0.0) == 0.5
    assert cosine_similarity_to_relevance_score(-0.2) == 0.4
    assert cosine_similarity_to_relevance_score(-0.8) == 0.1
    assert cosine_similarity_to_relevance_score(-1.0) == 0.0
    assert cosine_similarity_to_relevance_score(None) is None
    # Floating point precision overflow guards
    assert cosine_similarity_to_relevance_score(1.0000000002) == 1.0
    assert cosine_similarity_to_relevance_score(-1.0000000002) == 0.0


def test_search_preserves_negative_cosine_ranking_without_clamping(memory_store: VectorStore):
    """
    Verify search results with negative cosine similarities are not squashed to 0.0
    and preserve strictly distinct relative ranking and raw scores.
    """
    # Create two passages with negative projections against query [1.0, 0.01, 0.01, ...]
    # Point 1: slightly negative correlation (-0.2)
    # Point 2: strongly negative correlation (-0.8)
    dim = memory_store.vector_dim
    v1 = [0.0] * dim
    v1[0] = -0.2
    v1[1] = math.sqrt(1.0 - 0.2**2)

    v2 = [0.0] * dim
    v2[0] = -0.8
    v2[1] = math.sqrt(1.0 - 0.8**2)

    c1 = make_test_chunk(passage_id="neg_mild", text="Mildly distant guideline text.")
    c2 = make_test_chunk(passage_id="neg_strong", text="Strongly opposite guideline text.")

    memory_store.upsert_chunks([c1, c2], [v1, v2])

    query = [0.0] * dim
    query[0] = 1.0

    results = memory_store.search(query, top_k=2)
    assert len(results) == 2

    r1, r2 = results[0], results[1]
    assert r1.passage_id == "neg_mild"
    assert r2.passage_id == "neg_strong"

    # Crucial assertion: neither score is silently clamped to 0.0!
    assert r1.relevance_score > 0.0
    assert r2.relevance_score > 0.0
    assert r1.relevance_score > r2.relevance_score
    assert round(r1.relevance_score, 1) == 0.4  # (-0.2 + 1) / 2 = 0.4
    assert round(r2.relevance_score, 1) == 0.1  # (-0.8 + 1) / 2 = 0.1

    # Raw cosine scores must be preserved in retrieval_metadata
    assert r1.retrieval_metadata["raw_cosine_score"] < 0.0
    assert r2.retrieval_metadata["raw_cosine_score"] < 0.0
    assert r1.retrieval_metadata["raw_cosine_score"] > r2.retrieval_metadata["raw_cosine_score"]


# =====================================================================
# 6. Filtering (document_id and source_type)
# =====================================================================

def test_search_filtering_by_document_id(memory_store: VectorStore):
    """Verify filtering by document_id retrieves only matching passages."""
    c1 = make_test_chunk(passage_id="p1", document_id="DOC_NICE_NG28", text="NICE guideline on diabetes")
    c2 = make_test_chunk(passage_id="p2", document_id="DOC_ADA_2024", text="ADA guideline on diabetes")
    v1 = make_test_vector(primary_index=0)
    v2 = make_test_vector(primary_index=0)

    memory_store.upsert_chunks([c1, c2], [v1, v2])

    query = make_test_vector(primary_index=0)
    results = memory_store.search(query, document_id="DOC_ADA_2024")

    assert len(results) == 1
    assert results[0].passage_id == "p2"
    assert results[0].retrieval_metadata["document_id"] == "DOC_ADA_2024"


def test_search_filtering_by_source_type(memory_store: VectorStore):
    """Verify filtering by source_type retrieves only matching passages."""
    c1 = make_test_chunk(passage_id="p1", source_type="clinical_guideline")
    c2 = make_test_chunk(passage_id="p2", source_type="drug_label")
    v1 = make_test_vector(primary_index=10)
    v2 = make_test_vector(primary_index=10)

    memory_store.upsert_chunks([c1, c2], [v1, v2])

    query = make_test_vector(primary_index=10)
    results = memory_store.search(query, source_type="drug_label")

    assert len(results) == 1
    assert results[0].passage_id == "p2"
    assert results[0].retrieval_metadata["source_type"] == "drug_label"


def test_search_filtering_by_both_document_id_and_source_type(memory_store: VectorStore):
    """Verify compound filtering with both document_id and source_type."""
    c1 = make_test_chunk(passage_id="p1", document_id="DOC_1", source_type="guideline")
    c2 = make_test_chunk(passage_id="p2", document_id="DOC_1", source_type="protocol")
    c3 = make_test_chunk(passage_id="p3", document_id="DOC_2", source_type="guideline")
    v = make_test_vector(primary_index=0)

    memory_store.upsert_chunks([c1, c2, c3], [v, v, v])

    query = make_test_vector(primary_index=0)
    results = memory_store.search(query, document_id="DOC_1", source_type="guideline")

    assert len(results) == 1
    assert results[0].passage_id == "p1"


def test_search_filter_matching_nothing_returns_empty_list(memory_store: VectorStore):
    """Verify filtering that matches no documents safely returns empty list."""
    c = make_test_chunk(passage_id="p1", document_id="DOC_REAL")
    v = make_test_vector(primary_index=0)
    memory_store.upsert_chunks([c], [v])

    query = make_test_vector(primary_index=0)
    results = memory_store.search(query, document_id="NON_EXISTENT_DOC")
    assert results == []


def test_search_rejects_blank_filters(memory_store: VectorStore):
    """Verify empty or whitespace-only filter strings are rejected."""
    query = make_test_vector(primary_index=0)
    with pytest.raises(VectorStoreValidationError, match="document_id filter must be a non-empty string"):
        memory_store.search(query, document_id="   ")

    with pytest.raises(VectorStoreValidationError, match="source_type filter must be a non-empty string"):
        memory_store.search(query, source_type="")


# =====================================================================
# 7. Metadata Preservation & Payload Safety
# =====================================================================

def test_metadata_preservation_in_reference_passage(memory_store: VectorStore):
    """Verify all chunk and embedding provenance fields are preserved."""
    chunk = make_test_chunk(
        passage_id="NICE_2024_0042",
        document_id="NICE_GL_2024",
        text="Kidney disease staging and eGFR surveillance thresholds.",
        source_title="NICE Clinical Guideline 182",
        source_type="clinical_guideline",
        section="Section 2.1 - Renal Function Monitoring",
        chunk_index=42,
        publication_date="2024-03-20",
        metadata={"custom_guideline_key": "CKD_Stage_3"},
    )
    v = make_test_vector(primary_index=15)
    memory_store.upsert_chunks([chunk], [v])

    query = make_test_vector(primary_index=15)
    results = memory_store.search(query, top_k=1)

    assert len(results) == 1
    passage: ReferencePassage = results[0]

    assert passage.passage_id == "NICE_2024_0042"
    assert passage.title == "NICE Clinical Guideline 182 - Section 2.1 - Renal Function Monitoring"
    assert passage.content == "Kidney disease staging and eGFR surveillance thresholds."
    assert passage.source_document == "NICE Clinical Guideline 182"
    assert passage.section_or_page == "Section 2.1 - Renal Function Monitoring"
    assert passage.relevance_score is not None
    assert 0.0 <= passage.relevance_score <= 1.0

    meta = passage.retrieval_metadata
    assert meta["document_id"] == "NICE_GL_2024"
    assert meta["source_type"] == "clinical_guideline"
    assert meta["chunk_index"] == 42
    assert meta["publication_date"] == "2024-03-20"
    assert meta["custom_guideline_key"] == "CKD_Stage_3"
    assert "qdrant_point_id" in meta


def test_malformed_payload_skipped_safely(memory_store: VectorStore, memory_client: QdrantClient):
    """
    Verify that if a point with a malformed payload exists in Qdrant,
    search skips the malformed point without crashing or dropping valid points.
    """
    # 1. Insert a valid point through VectorStore
    valid_chunk = make_test_chunk(passage_id="valid_point", text="Valid clinical text.")
    valid_vec = make_test_vector(primary_index=0)
    memory_store.upsert_chunks([valid_chunk], [valid_vec])

    # 2. Directly inject a malformed point directly into Qdrant collection (missing text)
    bad_point_id = str(uuid.uuid4())
    memory_client.upsert(
        collection_name=memory_store.collection_name,
        points=[
            models.PointStruct(
                id=bad_point_id,
                vector=make_test_vector(primary_index=0),
                payload={"passage_id": "bad_point", "text": ""},  # empty text -> malformed
            )
        ],
    )

    query = make_test_vector(primary_index=0)
    results = memory_store.search(query, top_k=10)

    # Must return valid point and safely skip the malformed one
    assert len(results) == 1
    assert results[0].passage_id == "valid_point"


def test_empty_collection_search_returns_empty_list(memory_store: VectorStore):
    """Verify searching an empty collection returns empty list without error."""
    query = make_test_vector(primary_index=0)
    results = memory_store.search(query)
    assert results == []


# =====================================================================
# 8. Count Operations
# =====================================================================

def test_count_total_and_filtered(memory_store: VectorStore):
    """Verify count method returns correct counts for total and filtered points."""
    assert memory_store.count() == 0

    c1 = make_test_chunk(passage_id="p1", document_id="doc1", source_type="guideline")
    c2 = make_test_chunk(passage_id="p2", document_id="doc1", source_type="drug_label")
    c3 = make_test_chunk(passage_id="p3", document_id="doc2", source_type="guideline")
    v = make_test_vector(primary_index=0)

    memory_store.upsert_chunks([c1, c2, c3], [v, v, v])

    assert memory_store.count() == 3
    assert memory_store.count(document_id="doc1") == 2
    assert memory_store.count(document_id="doc2") == 1
    assert memory_store.count(source_type="guideline") == 2
    assert memory_store.count(source_type="drug_label") == 1
    assert memory_store.count(document_id="doc1", source_type="guideline") == 1
    assert memory_store.count(document_id="doc_missing") == 0


def test_count_rejects_invalid_filters(memory_store: VectorStore):
    """Verify count rejects blank or non-string filters."""
    with pytest.raises(VectorStoreValidationError, match="document_id filter must be a non-empty string"):
        memory_store.count(document_id="   ")

    with pytest.raises(VectorStoreValidationError, match="source_type filter must be a non-empty string"):
        memory_store.count(source_type="")


# =====================================================================
# 9. Collection Lifecycle & Context Manager
# =====================================================================

def test_delete_collection(memory_store: VectorStore, memory_client: QdrantClient):
    """Verify delete_collection drops collection from Qdrant."""
    assert memory_client.collection_exists(memory_store.collection_name) is True
    memory_store.delete_collection()
    assert memory_client.collection_exists(memory_store.collection_name) is False


def test_context_manager_support():
    """Verify VectorStore can be used cleanly as a context manager."""
    client = QdrantClient(":memory:")
    with VectorStore(client=client, collection_name="ctx_test") as store:
        assert store.collection_name == "ctx_test"
        assert client.collection_exists("ctx_test") is True


# =====================================================================
# 10. Privacy & Sanitized Logging Tests
# =====================================================================

def test_sensitive_clinical_text_not_logged(caplog, memory_store: VectorStore):
    """
    Verify that raw clinical text, patient-sensitive strings, or query vector numbers
    are NEVER logged by the VectorStore service.
    """
    secret_text = "CONFIDENTIAL_PATIENT_CLINICAL_DIAGNOSIS_DATA_DO_NOT_LOG"
    chunk = make_test_chunk(passage_id="p_secret", text=secret_text)
    vector = make_test_vector(primary_index=0)

    with caplog.at_level(logging.DEBUG):
        memory_store.upsert_chunks([chunk], [vector])
        query = make_test_vector(primary_index=0)
        results = memory_store.search(query, top_k=1)
        assert len(results) == 1

    # Verify sensitive phrase never appears in any log records
    assert secret_text not in caplog.text


def test_get_by_id_sanitized_logging_on_reconstruction_failure(
    caplog, memory_store: VectorStore, memory_client: QdrantClient
):
    """
    Verify get_by_id logs only the exception class name and safe point ID
    when payload reconstruction fails, avoiding any raw exception messages.
    """
    secret_marker = "SECRET_CLINICAL_MARKER_DO_NOT_EXPOSE"
    bad_id = str(uuid.uuid4())

    # Directly inject point with invalid payload (empty text will fail ReferencePassage validation)
    memory_client.upsert(
        collection_name=memory_store.collection_name,
        points=[
            models.PointStruct(
                id=bad_id,
                vector=make_test_vector(primary_index=0),
                payload={
                    "passage_id": bad_id,
                    "text": "   ",  # invalid whitespace
                    "source_title": secret_marker,
                },
            )
        ],
    )

    with caplog.at_level(logging.WARNING):
        result = memory_store.get_by_id(bad_id)
        assert result is None

    # Secret phrase must not appear in any log output
    assert secret_marker not in caplog.text
    # Exception type must be logged safely
    assert "ValidationError" in caplog.text


def test_get_by_id_sanitized_logging_on_retrieval_error(caplog, memory_store: VectorStore):
    """
    Verify get_by_id logs only the exception class name on retrieval errors,
    preventing any sensitive API error strings or tokens from leaking.
    """
    sensitive_token = "SENSITIVE_BEARER_TOKEN_INTERNAL_SECRET"
    with mock.patch.object(
        memory_store.client, "retrieve", side_effect=RuntimeError(sensitive_token)
    ):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(VectorStoreError):
                memory_store.get_by_id("valid_passage_id")

    assert sensitive_token not in caplog.text
    assert "RuntimeError" in caplog.text

