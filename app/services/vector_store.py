"""
MediTrace - Vector Store Service for Clinical RAG Pipelines
Provides a local or in-memory vector database service backed by Qdrant for storing
embedded reference document chunks and retrieving relevant clinical guideline passages.

CLINICAL SAFETY NOTICE:
-----------------------
MediTrace is an educational clinical documentation prototype.
It must NOT autonomously diagnose, prescribe, or make treatment decisions.
Dense vector embeddings and retrieved reference passages are for semantic search
and guideline retrieval only. All retrieved evidence and downstream clinical
alerts require independent verification by a licensed clinician.

ARCHITECTURE AND SPECIFICATIONS:
--------------------------------
- Vector Engine: Qdrant via official `qdrant-client` library
- Default Collection: clinical_guidelines
- Default Vector Dimension: 768 (matching gemini-embedding-001 output dimensionality)
- Distance Metric: Cosine similarity (models.Distance.COSINE)
- Storage: Dedicated local directory (defaults to qdrant_storage/) or :memory:
- Separation of Concerns:
    - Pure vector database operations (upsert, search, count, delete).
    - Accepts pre-computed query vectors and ChunkEmbedding objects.
    - Never invokes Gemini API or generates embeddings internally.
    - Preserves verbatim passage IDs and provenance metadata in payload.
    - Never logs raw clinical text, API keys, or sensitive request parameters.
"""

import logging
import math
import os
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from app.schemas import DocumentChunk, ReferencePassage
from app.services.embedding_service import ChunkEmbedding

logger = logging.getLogger(__name__)

# Default vector store configuration
DEFAULT_COLLECTION_NAME: str = "clinical_guidelines"
DEFAULT_STORAGE_PATH: str = "qdrant_storage"
DEFAULT_VECTOR_DIMENSION: int = 768


# =====================================================================
# Service Exceptions
# =====================================================================

class VectorStoreError(Exception):
    """Base exception for all MediTrace vector store operations."""
    pass


class VectorStoreConfigurationError(VectorStoreError, ValueError):
    """Raised when vector store configuration (paths, client, dimensions) is invalid."""
    pass


class VectorStoreValidationError(VectorStoreError, ValueError):
    """Raised when input vectors, IDs, or chunks are malformed or invalid."""
    pass


class VectorStoreInsertionError(VectorStoreError):
    """Raised when upserting vectors or payloads into Qdrant fails."""
    pass


class VectorStoreSearchError(VectorStoreError):
    """Raised when searching the vector database fails."""
    pass


# =====================================================================
# Helper Functions
# =====================================================================

def passage_id_to_point_id(passage_id: Union[str, int, uuid.UUID]) -> str:
    """
    Convert an arbitrary passage ID into a deterministic valid Qdrant point ID.
    Qdrant requires point IDs to be unsigned 64-bit integers or valid UUID strings.

    If passage_id is already a valid UUID string or int, it is formatted cleanly.
    Otherwise, a deterministic UUIDv5 in the DNS namespace is generated.
    The original passage_id string is always preserved in the point payload.
    """
    if isinstance(passage_id, bool):
        raise VectorStoreValidationError("Passage ID must not be a boolean value.")

    if isinstance(passage_id, int):
        if passage_id < 0:
            raise VectorStoreValidationError(f"Integer point ID must be non-negative: {passage_id}")
        return str(passage_id)

    if isinstance(passage_id, uuid.UUID):
        return str(passage_id)

    if not isinstance(passage_id, str) or not passage_id.strip():
        raise VectorStoreValidationError("Passage ID must be a non-empty string or positive integer.")

    cleaned_id = passage_id.strip()

    # Check if already a valid UUID hex
    try:
        parsed_uuid = uuid.UUID(cleaned_id)
        return str(parsed_uuid)
    except (ValueError, AttributeError):
        # Deterministically derive UUIDv5 from cleaned_id
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, cleaned_id))


def cosine_similarity_to_relevance_score(cosine_score: Optional[float]) -> Optional[float]:
    """
    Convert a raw cosine similarity score in [-1.0, 1.0] to a normalized relevance score in [0.0, 1.0].

    RATIONALE & SCHEMA SPECIFICATION:
    ---------------------------------
    The ReferencePassage schema enforces: `relevance_score: Optional[float] = Field(ge=0.0, le=1.0)`.
    Cosine similarity naturally ranges from -1.0 (diametrically opposite orientation) through
    0.0 (orthogonal / neutral) to +1.0 (identical orientation).

    Rather than silently clamping negative scores to 0.0 (which destroys relative ranking among
    negative passages by squashing distinct negative values like -0.2 and -0.8 to the same 0.0),
    we apply linear normalization:

        normalized_score = (cosine_score + 1.0) / 2.0

    Characteristics:
      - Strictly monotonic: Preserves exact relative ranking across all possible cosine values.
      - 1.0 (identical vectors)     -> 1.0
      - 0.0 (orthogonal / neutral)   -> 0.5
      - -0.2 (weak dissimilarity)    -> 0.4
      - -0.8 (strong dissimilarity)  -> 0.1
      - -1.0 (opposite orientation)  -> 0.0
      - Float guard: clamps any micro-overflow due to floating point inaccuracies to [0.0, 1.0].
      - Bounded and rounded to 4 decimal places for clean, deterministic serialization.

    The verbatim Qdrant cosine similarity is also preserved in `retrieval_metadata["raw_cosine_score"]`.
    """
    if cosine_score is None:
        return None

    try:
        raw = float(cosine_score)
    except (ValueError, TypeError):
        return None

    # Guard against float rounding exceeding [-1.0, 1.0]
    safe_raw = max(-1.0, min(1.0, raw))
    normalized = (safe_raw + 1.0) / 2.0
    return round(normalized, 4)


# =====================================================================
# Vector Store Implementation
# =====================================================================

class VectorStore:
    """
    Qdrant-backed vector storage service for indexing reference document chunks
    and retrieving relevant clinical guideline passages using cosine similarity.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        storage_path: Optional[str] = DEFAULT_STORAGE_PATH,
        client: Optional[QdrantClient] = None,
        vector_dim: int = DEFAULT_VECTOR_DIMENSION,
        auto_ensure_collection: bool = True,
        **client_kwargs: Any,
    ):
        """
        Initialize the Qdrant vector store service.

        Args:
            collection_name: Target collection name (default: "clinical_guidelines").
            storage_path: Local directory path for storage files (default: "qdrant_storage"),
                          or ":memory:" for transient in-memory operations.
            client: Optional injected QdrantClient instance (useful for testing or external servers).
            vector_dim: Expected embedding dimensionality (default: 768).
            auto_ensure_collection: If True, automatically creates/verifies the collection on init.
            **client_kwargs: Additional parameters forwarded to QdrantClient if initializing.
        """
        # Validate collection name
        if not collection_name or not isinstance(collection_name, str) or not collection_name.strip():
            raise VectorStoreConfigurationError("Collection name must be a non-empty string.")
        self.collection_name = collection_name.strip()

        # Validate vector dimension (rejecting bool explicitly)
        if isinstance(vector_dim, bool) or not isinstance(vector_dim, int) or vector_dim <= 0:
            raise VectorStoreConfigurationError(
                f"Vector dimension must be a positive integer (not bool), got {type(vector_dim).__name__}: {vector_dim}"
            )
        self.vector_dim = vector_dim

        # Initialize Qdrant client
        if client is not None:
            self.client = client
            self.storage_path = None
        else:
            resolved_path = storage_path if storage_path is not None else DEFAULT_STORAGE_PATH
            self.storage_path = resolved_path
            if resolved_path == ":memory:":
                self.client = QdrantClient(":memory:", **client_kwargs)
            else:
                try:
                    os.makedirs(resolved_path, exist_ok=True)
                    self.client = QdrantClient(path=resolved_path, **client_kwargs)
                except Exception as exc:
                    logger.error("Failed to initialize QdrantClient at path '%s'", resolved_path)
                    raise VectorStoreConfigurationError(
                        f"Failed to initialize local Qdrant storage at '{resolved_path}': {type(exc).__name__}"
                    ) from exc

        if auto_ensure_collection:
            self.ensure_collection()

    def ensure_collection(self, force_recreate: bool = False) -> None:
        """
        Ensure the target collection exists with the required vector dimension and cosine similarity.
        Validates both vector dimension and cosine distance configuration if the collection already exists.

        Args:
            force_recreate: If True, deletes any existing collection and recreates it.
        """
        try:
            exists = self.client.collection_exists(self.collection_name)
        except Exception as exc:
            logger.error("Failed checking existence of collection '%s'", self.collection_name)
            raise VectorStoreConfigurationError(
                f"Failed checking collection existence: {type(exc).__name__}"
            ) from exc

        if exists and force_recreate:
            logger.info("Recreating collection '%s'", self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if exists:
            # Verify dimensionality and distance metric on existing collection
            try:
                info = self.client.get_collection(self.collection_name)
                existing_dim = None
                existing_distance = None
                vectors_cfg = getattr(info.config.params, "vectors", None)
                if hasattr(vectors_cfg, "size"):
                    existing_dim = vectors_cfg.size
                    existing_distance = getattr(vectors_cfg, "distance", None)
                elif isinstance(vectors_cfg, dict) and "" in vectors_cfg:
                    existing_dim = getattr(vectors_cfg[""], "size", None)
                    existing_distance = getattr(vectors_cfg[""], "distance", None)

                # 1. Validate vector dimension
                if existing_dim is not None and existing_dim != self.vector_dim:
                    raise VectorStoreConfigurationError(
                        f"Collection '{self.collection_name}' exists with vector dimension {existing_dim}, "
                        f"which does not match configured dimension {self.vector_dim}."
                    )

                # 2. Validate cosine distance metric
                if existing_distance is not None and existing_distance != models.Distance.COSINE:
                    raise VectorStoreConfigurationError(
                        f"Collection '{self.collection_name}' exists with distance metric '{existing_distance}', "
                        f"which does not match expected metric '{models.Distance.COSINE}'."
                    )
            except VectorStoreConfigurationError:
                raise
            except Exception as exc:
                logger.warning(
                    "Could not inspect vector parameters on existing collection '%s': %s",
                    self.collection_name,
                    type(exc).__name__,
                )
        else:
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
                logger.info(
                    "Created collection '%s' with dimension %d and cosine distance",
                    self.collection_name,
                    self.vector_dim,
                )
            except Exception as exc:
                logger.error("Failed creating collection '%s'", self.collection_name)
                raise VectorStoreConfigurationError(
                    f"Failed to create collection '{self.collection_name}': {type(exc).__name__}"
                ) from exc

    def _validate_vector(self, vector: Any, context: str = "vector") -> List[float]:
        """
        Validate vector type, dimensionality, and finiteness.
        Never logs vector values or query content.
        """
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        if not isinstance(vector, (list, tuple)):
            raise VectorStoreValidationError(
                f"Expected {context} to be a list of floats, got {type(vector).__name__}."
            )

        if len(vector) != self.vector_dim:
            raise VectorStoreValidationError(
                f"Dimension mismatch for {context}: expected {self.vector_dim}, got {len(vector)}."
            )

        float_vector: List[float] = []
        for idx, val in enumerate(vector):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise VectorStoreValidationError(
                    f"Element at index {idx} in {context} is not a valid numeric float."
                )
            fval = float(val)
            if not math.isfinite(fval):
                raise VectorStoreValidationError(
                    f"Element at index {idx} in {context} is non-finite ({fval})."
                )
            float_vector.append(fval)

        return float_vector

    def upsert_embeddings(self, chunk_embeddings: List[ChunkEmbedding]) -> int:
        """
        Insert embedded DocumentChunk objects into Qdrant.
        Preserves original passage ID and source metadata in each point's payload.

        Args:
            chunk_embeddings: List of ChunkEmbedding objects containing vectors and DocumentChunks.

        Returns:
            Number of points successfully upserted.
        """
        if not isinstance(chunk_embeddings, list) or not chunk_embeddings:
            raise VectorStoreValidationError("chunk_embeddings must be a non-empty list of ChunkEmbedding instances.")

        points: List[models.PointStruct] = []

        for idx, item in enumerate(chunk_embeddings):
            if not isinstance(item, ChunkEmbedding):
                raise VectorStoreValidationError(
                    f"Item at index {idx} must be an instance of ChunkEmbedding, got {type(item).__name__}."
                )

            chunk = item.chunk
            if not isinstance(chunk, DocumentChunk):
                raise VectorStoreValidationError(
                    f"Item at index {idx} has invalid chunk attribute: expected DocumentChunk."
                )

            # Validate vector dimensions and finiteness
            clean_vector = self._validate_vector(item.embedding, f"chunk '{chunk.passage_id}' embedding")

            # Deterministic Qdrant point ID
            point_id = passage_id_to_point_id(chunk.passage_id)

            # Build preserved payload
            combined_metadata = {**chunk.metadata, **(item.metadata or {})}
            payload: Dict[str, Any] = {
                "passage_id": chunk.passage_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "source_title": chunk.source_title,
                "source_type": chunk.source_type,
                "section": chunk.section,
                "publication_date": chunk.publication_date,
                "chunk_index": chunk.chunk_index,
                "metadata": combined_metadata,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=clean_vector,
                    payload=payload,
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(
                "Successfully upserted %d points into collection '%s'",
                len(points),
                self.collection_name,
            )
            return len(points)
        except Exception as exc:
            logger.error(
                "Failed to upsert %d points into collection '%s': %s",
                len(points),
                self.collection_name,
                type(exc).__name__,
            )
            raise VectorStoreInsertionError(
                f"Failed to upsert points into collection '{self.collection_name}': {type(exc).__name__}"
            ) from exc

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Convenience method to insert DocumentChunk objects with parallel pre-computed embeddings.

        Args:
            chunks: List of DocumentChunk objects.
            embeddings: Parallel list of embedding vectors matching chunks.

        Returns:
            Number of points successfully upserted.
        """
        if not isinstance(chunks, list) or not chunks:
            raise VectorStoreValidationError("chunks must be a non-empty list of DocumentChunk instances.")
        if not isinstance(embeddings, list) or not embeddings:
            raise VectorStoreValidationError("embeddings must be a non-empty list of vectors.")
        if len(chunks) != len(embeddings):
            raise VectorStoreValidationError(
                f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        chunk_embeddings: List[ChunkEmbedding] = []
        for c, emb in zip(chunks, embeddings):
            chunk_embeddings.append(
                ChunkEmbedding(
                    passage_id=c.passage_id,
                    embedding=emb,
                    chunk=c,
                    metadata=c.metadata,
                )
            )

        return self.upsert_embeddings(chunk_embeddings)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None,
        source_type: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[ReferencePassage]:
        """
        Perform similarity search using an already-generated query vector.
        Does NOT invoke Gemini or any external embedding model.

        Args:
            query_vector: Precomputed dense embedding vector of dimension 768.
            top_k: Maximum number of relevant passages to retrieve (default: 5).
            document_id: Optional filter by specific parent document identifier.
            source_type: Optional filter by source classification (e.g., "clinical_guideline").
            score_threshold: Optional minimum cosine similarity score threshold.

        Returns:
            List of ReferencePassage objects ranked by cosine similarity score.
        """
        # Validate query vector
        clean_query = self._validate_vector(query_vector, context="query_vector")

        # Validate top_k (rejecting bool explicitly)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise VectorStoreValidationError(
                f"top_k must be a positive integer (not bool), got {type(top_k).__name__}: {top_k}"
            )

        # Construct optional filters
        must_conditions: List[models.FieldCondition] = []
        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise VectorStoreValidationError("document_id filter must be a non-empty string.")
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id.strip()),
                )
            )

        if source_type is not None:
            if not isinstance(source_type, str) or not source_type.strip():
                raise VectorStoreValidationError("source_type filter must be a non-empty string.")
            must_conditions.append(
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value=source_type.strip()),
                )
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        try:
            # Query Qdrant using the version-appropriate client API
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=clean_query,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                scored_points = response.points if hasattr(response, "points") else response
            elif hasattr(self.client, "search"):
                scored_points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=clean_query,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                raise VectorStoreSearchError("QdrantClient provides neither query_points nor search method.")
        except Exception as exc:
            if isinstance(exc, (VectorStoreValidationError, VectorStoreSearchError)):
                raise
            logger.error(
                "Failed to execute search on collection '%s': %s",
                self.collection_name,
                type(exc).__name__,
            )
            raise VectorStoreSearchError(
                f"Vector search failed on collection '{self.collection_name}': {type(exc).__name__}"
            ) from exc

        # Handle empty results safely
        if not scored_points:
            return []

        results: List[ReferencePassage] = []
        for point in scored_points:
            payload = point.payload or {}
            try:
                passage_id = payload.get("passage_id") or str(point.id)
                text = payload.get("text") or payload.get("content")
                source_title = payload.get("source_title") or payload.get("source_document") or "Reference Document"
                source_type_val = payload.get("source_type", "reference")
                doc_id = payload.get("document_id", "unknown")
                section = payload.get("section") or payload.get("section_or_page")
                pub_date = payload.get("publication_date")
                chunk_index = payload.get("chunk_index", 0)
                metadata = payload.get("metadata") or {}

                if not text or not str(text).strip():
                    logger.warning("Skipping point '%s' due to missing text content", point.id)
                    continue

                # Convert cosine similarity [-1.0, 1.0] to normalized relevance score [0.0, 1.0].
                # Preserves linear ranking across all similarity ranges without silently clamping.
                normalized_score = cosine_similarity_to_relevance_score(point.score)

                title_with_section = f"{source_title} - {section}" if section else source_title
                retrieval_meta = {
                    **metadata,
                    "document_id": doc_id,
                    "source_type": source_type_val,
                    "chunk_index": chunk_index,
                    "publication_date": pub_date,
                    "qdrant_point_id": str(point.id),
                }
                if point.score is not None:
                    try:
                        retrieval_meta["raw_cosine_score"] = round(float(point.score), 6)
                    except (ValueError, TypeError):
                        pass

                ref_passage = ReferencePassage(
                    passage_id=passage_id,
                    title=title_with_section,
                    content=str(text).strip(),
                    source_document=source_title,
                    section_or_page=section,
                    relevance_score=normalized_score,
                    retrieval_metadata=retrieval_meta,
                )
                results.append(ref_passage)

            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping search result point '%s' with malformed payload: %s",
                    point.id,
                    type(exc).__name__,
                )
                continue

        return results

    def count(
        self,
        document_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> int:
        """
        Count total or filtered points stored in the collection.

        Args:
            document_id: Optional filter by specific parent document identifier.
            source_type: Optional filter by source classification.

        Returns:
            Integer count of matching points.
        """
        must_conditions: List[models.FieldCondition] = []
        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise VectorStoreValidationError("document_id filter must be a non-empty string.")
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id.strip()),
                )
            )

        if source_type is not None:
            if not isinstance(source_type, str) or not source_type.strip():
                raise VectorStoreValidationError("source_type filter must be a non-empty string.")
            must_conditions.append(
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value=source_type.strip()),
                )
            )

        count_filter = models.Filter(must=must_conditions) if must_conditions else None

        try:
            res = self.client.count(
                collection_name=self.collection_name,
                count_filter=count_filter,
                exact=True,
            )
            return int(res.count)
        except Exception as exc:
            logger.error("Failed to count points in collection '%s'", self.collection_name)
            raise VectorStoreError(
                f"Vector store count operation failed: {type(exc).__name__}"
            ) from exc

    def get_by_id(self, passage_id: str) -> Optional[ReferencePassage]:
        """
        Retrieve a single reference passage by its original passage ID.

        Args:
            passage_id: The passage ID of the chunk to retrieve.

        Returns:
            ReferencePassage if found, or None.
        """
        point_id = passage_id_to_point_id(passage_id)
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.error(
                "Failed retrieving point '%s' from collection '%s': %s",
                point_id,
                self.collection_name,
                type(exc).__name__,
            )
            raise VectorStoreError(
                f"Failed retrieving point '{point_id}': {type(exc).__name__}"
            ) from exc

        if not records:
            return None

        point = records[0]
        payload = point.payload or {}
        try:
            pid = payload.get("passage_id") or str(point.id)
            text = payload.get("text") or payload.get("content") or ""
            source_title = payload.get("source_title") or payload.get("source_document") or "Reference Document"
            source_type_val = payload.get("source_type", "reference")
            doc_id = payload.get("document_id", "unknown")
            section = payload.get("section") or payload.get("section_or_page")
            pub_date = payload.get("publication_date")
            chunk_index = payload.get("chunk_index", 0)
            metadata = payload.get("metadata") or {}

            title_with_section = f"{source_title} - {section}" if section else source_title
            return ReferencePassage(
                passage_id=pid,
                title=title_with_section,
                content=str(text).strip(),
                source_document=source_title,
                section_or_page=section,
                relevance_score=1.0,
                retrieval_metadata={
                    **metadata,
                    "document_id": doc_id,
                    "source_type": source_type_val,
                    "chunk_index": chunk_index,
                    "publication_date": pub_date,
                    "qdrant_point_id": str(point.id),
                },
            )
        except (ValidationError, KeyError, TypeError, ValueError, Exception) as exc:
            logger.warning(
                "Failed reconstructing ReferencePassage for point '%s': %s",
                point.id,
                type(exc).__name__,
            )
            return None

    def delete_collection(self) -> None:
        """Delete the collection if it exists."""
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                logger.info("Deleted collection '%s'", self.collection_name)
        except Exception as exc:
            logger.error("Failed deleting collection '%s'", self.collection_name)
            raise VectorStoreError(f"Failed deleting collection: {type(exc).__name__}") from exc

    def close(self) -> None:
        """Close the underlying client connection if open."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
