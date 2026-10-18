"""
MediTrace - RAG Evidence Retrieval Service
Coordinates the dense embedding service and the Qdrant vector store to retrieve
evidence-based clinical guideline passages and literature excerpts.

CLINICAL SAFETY NOTICE:
-----------------------
MediTrace is an educational clinical documentation prototype.
It must NOT autonomously diagnose, prescribe, or make treatment decisions.
All retrieved reference passages and semantic similarities are for clinician
information and evidence verification only.

PROMPT INJECTION & UNTRUSTED DATA GUARD:
----------------------------------------
Retrieved passages from guidelines or literature are passive reference material,
NOT executable instructions. Downstream LLM pipelines must never execute instructions
contained within retrieved passages.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from app.schemas import ReferencePassage
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.vector_store import (
    VectorStore,
    VectorStoreError,
    cosine_similarity_to_relevance_score,
)

logger = logging.getLogger(__name__)

# Default retrieval configuration
DEFAULT_RETRIEVAL_TOP_K: int = 5


# =====================================================================
# Similarity Score Conversion Helpers
# =====================================================================

def relevance_score_to_cosine_similarity(relevance_score: Optional[float]) -> Optional[float]:
    """
    Convert a normalized relevance score in [0.0, 1.0] to a raw cosine similarity score in [-1.0, 1.0].

    MATHEMATICAL SPECIFICATION:
    ---------------------------
    This function is the exact inverse of cosine_similarity_to_relevance_score() from vector_store.py:
        relevance_score = (cosine_similarity + 1.0) / 2.0
    Therefore:
        cosine_similarity = 2.0 * relevance_score - 1.0

    Mapping:
      - 1.0 relevance score -> +1.0 cosine similarity (identical vectors)
      - 0.5 relevance score ->  0.0 cosine similarity (orthogonal vectors)
      - 0.4 relevance score -> -0.2 cosine similarity
      - 0.1 relevance score -> -0.8 cosine similarity
      - 0.0 relevance score -> -1.0 cosine similarity (diametrically opposite)

    Guards:
      - Bounded strictly within [-1.0, 1.0].
      - Rounded to 4 decimal places for consistency with cosine_similarity_to_relevance_score().
    """
    if relevance_score is None:
        return None

    try:
        rel = float(relevance_score)
    except (ValueError, TypeError):
        return None

    if not math.isfinite(rel):
        return None

    # Guard and clamp to valid normalized range [0.0, 1.0]
    safe_rel = max(0.0, min(1.0, rel))
    raw = 2.0 * safe_rel - 1.0
    return round(raw, 4)


def validate_dimension_compatibility(
    embedding_service: Optional[Any],
    vector_store: Optional[Any],
) -> None:
    """
    Verify that the embedding service output dimension matches the vector store dimension.

    Raises:
        RetrievalConfigurationError: If both services have dimension attributes that do not match.
    """
    if embedding_service is None or vector_store is None:
        return

    raw_emb = getattr(embedding_service, "output_dimensionality", None)
    if not isinstance(raw_emb, int) or isinstance(raw_emb, bool):
        raw_emb = getattr(embedding_service, "vector_dim", None)
    emb_dim = raw_emb if (isinstance(raw_emb, int) and not isinstance(raw_emb, bool)) else None

    raw_store = getattr(vector_store, "vector_dim", None)
    store_dim = raw_store if (isinstance(raw_store, int) and not isinstance(raw_store, bool)) else None

    if emb_dim is not None and store_dim is not None and emb_dim != store_dim:
        raise RetrievalConfigurationError(
            f"Embedding service output dimension ({emb_dim}) does not match "
            f"vector store dimension ({store_dim})."
        )


# =====================================================================
# Service Exceptions
# =====================================================================

class RetrievalServiceError(Exception):
    """Base exception for all MediTrace retrieval service operations."""
    pass


class RetrievalConfigurationError(RetrievalServiceError, ValueError):
    """Raised when retrieval service dependencies or configuration are invalid."""
    pass


class RetrievalValidationError(RetrievalServiceError, ValueError):
    """Raised when query or retrieval search parameters are invalid."""
    pass


class RetrievalEmbeddingError(RetrievalServiceError):
    """Raised when query embedding generation fails."""
    pass


class RetrievalVectorStoreError(RetrievalServiceError):
    """Raised when vector database search operations fail."""
    pass


# =====================================================================
# Retrieval Service Implementation
# =====================================================================

class RetrievalService:
    """
    RAG evidence retrieval service for MediTrace clinical copilot.
    Orchestrates query embedding generation and similarity search across
    indexed reference guideline passages.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        default_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        default_similarity_threshold: Optional[float] = None,
    ):
        """
        Initialize the retrieval service with dependency injection.

        Args:
            embedding_service: Optional injected EmbeddingService instance.
            vector_store: Optional injected VectorStore instance.
            default_top_k: Default maximum number of passages to retrieve (default: 5).
            default_similarity_threshold: Optional default minimum relevance score threshold [0.0, 1.0].
        """
        # Validate dependencies if explicitly provided
        if embedding_service is not None and not hasattr(embedding_service, "embed_query"):
            raise RetrievalConfigurationError(
                "embedding_service must provide a callable embed_query method."
            )
        if vector_store is not None and not hasattr(vector_store, "search"):
            raise RetrievalConfigurationError(
                "vector_store must provide a callable search method."
            )

        # Validate vector dimension compatibility between provided services
        validate_dimension_compatibility(embedding_service, vector_store)

        # Validate default_top_k (rejecting bool explicitly)
        if isinstance(default_top_k, bool) or not isinstance(default_top_k, int) or default_top_k <= 0:
            raise RetrievalConfigurationError(
                f"default_top_k must be a positive integer (not bool), got {type(default_top_k).__name__}"
            )

        # Validate default_similarity_threshold
        if default_similarity_threshold is not None:
            if isinstance(default_similarity_threshold, bool) or not isinstance(
                default_similarity_threshold, (int, float)
            ):
                raise RetrievalConfigurationError(
                    f"default_similarity_threshold must be a numeric float, got {type(default_similarity_threshold).__name__}"
                )
            f_thresh = float(default_similarity_threshold)
            if not math.isfinite(f_thresh) or f_thresh < 0.0 or f_thresh > 1.0:
                raise RetrievalConfigurationError(
                    f"default_similarity_threshold must be between 0.0 and 1.0, got {f_thresh}"
                )

        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_similarity_threshold = default_similarity_threshold

    @property
    def embedding_service(self) -> EmbeddingService:
        """Access or lazily initialize the embedding service."""
        if self._embedding_service is None:
            try:
                self._embedding_service = EmbeddingService()
            except Exception as exc:
                raise RetrievalConfigurationError(
                    f"Failed to initialize default EmbeddingService: {type(exc).__name__}"
                ) from exc
        return self._embedding_service

    @property
    def vector_store(self) -> VectorStore:
        """Access or lazily initialize the vector store service."""
        if self._vector_store is None:
            try:
                self._vector_store = VectorStore()
            except Exception as exc:
                raise RetrievalConfigurationError(
                    f"Failed to initialize default VectorStore: {type(exc).__name__}"
                ) from exc
        return self._vector_store

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[ReferencePassage]:
        """
        Retrieve relevant reference guideline passages for a clinical query.

        Args:
            query: Non-empty search term or clinical context string.
            top_k: Maximum number of passages to return (defaults to self.default_top_k).
            similarity_threshold: Optional minimum relevance score in [0.0, 1.0] (defaults to default_similarity_threshold).
            document_id: Optional filter by specific parent document identifier.
            source_type: Optional filter by source classification (e.g. 'clinical_guideline').

        Returns:
            List of ReferencePassage objects ordered by relevance score descending.
        """
        # 1. Validate query
        if query is None or not isinstance(query, str) or not query.strip():
            raise RetrievalValidationError("Query must be a non-empty string.")
        cleaned_query = query.strip()

        # 2. Validate top_k
        resolved_top_k = top_k if top_k is not None else self.default_top_k
        if isinstance(resolved_top_k, bool) or not isinstance(resolved_top_k, int) or resolved_top_k <= 0:
            raise RetrievalValidationError(
                f"top_k must be a positive integer (not bool), got {type(resolved_top_k).__name__}"
            )

        # 3. Validate similarity_threshold
        resolved_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.default_similarity_threshold
        )
        if resolved_threshold is not None:
            if isinstance(resolved_threshold, bool) or not isinstance(
                resolved_threshold, (int, float)
            ):
                raise RetrievalValidationError(
                    f"similarity_threshold must be a numeric float, got {type(resolved_threshold).__name__}"
                )
            f_thresh = float(resolved_threshold)
            if not math.isfinite(f_thresh) or f_thresh < 0.0 or f_thresh > 1.0:
                raise RetrievalValidationError(
                    f"similarity_threshold must be a finite float between 0.0 and 1.0, got {f_thresh}"
                )
            resolved_threshold = f_thresh

        # 4. Validate metadata filters
        if document_id is not None:
            if isinstance(document_id, bool) or not isinstance(document_id, str) or not document_id.strip():
                raise RetrievalValidationError("document_id filter must be a non-empty string.")
            document_id = document_id.strip()

        if source_type is not None:
            if isinstance(source_type, bool) or not isinstance(source_type, str) or not source_type.strip():
                raise RetrievalValidationError("source_type filter must be a non-empty string.")
            source_type = source_type.strip()

        # 5. Validate dimension compatibility of resolved services
        validate_dimension_compatibility(self.embedding_service, self.vector_store)

        # 6. Generate query embedding
        # Notice: Never log the query text itself to maintain privacy
        logger.debug("Generating embedding for retrieval query")
        try:
            query_vector = self.embedding_service.embed_query(cleaned_query)
        except Exception as exc:
            logger.error(
                "Failed to generate query embedding during retrieval: %s",
                type(exc).__name__,
            )
            raise RetrievalEmbeddingError(
                f"Failed to generate query embedding: {type(exc).__name__}"
            ) from exc

        # 7. Validate vector dimension against vector store configuration
        store_dim = getattr(self.vector_store, "vector_dim", None)
        if store_dim is not None and len(query_vector) != store_dim:
            raise RetrievalConfigurationError(
                f"Generated query vector dimension ({len(query_vector)}) does not match "
                f"vector store dimension ({store_dim})."
            )

        # 8. Map normalized threshold [0.0, 1.0] to raw cosine threshold [-1.0, 1.0] for Qdrant
        # Boundary & Consistency Guard:
        # Subtracting an epsilon (1e-4) prevents Qdrant's raw pre-filter from prematurely dropping
        # borderline candidates whose rounded relevance score satisfies resolved_threshold.
        # If resolved_threshold is 0.0, raw_score_threshold remains None (all passages match >= 0.0).
        raw_score_threshold: Optional[float] = None
        if resolved_threshold is not None and resolved_threshold > 0.0:
            raw_target = relevance_score_to_cosine_similarity(resolved_threshold)
            if raw_target is not None:
                raw_score_threshold = max(-1.0, round(raw_target - 1e-4, 4))

        # 9. Execute vector store search
        logger.debug("Searching vector store with top_k=%d", resolved_top_k)
        try:
            passages = self.vector_store.search(
                query_vector=query_vector,
                top_k=resolved_top_k,
                document_id=document_id,
                source_type=source_type,
                score_threshold=raw_score_threshold,
            )
        except Exception as exc:
            logger.error(
                "Failed to execute vector search during retrieval: %s",
                type(exc).__name__,
            )
            raise RetrievalVectorStoreError(
                f"Failed to search vector store: {type(exc).__name__}"
            ) from exc

        # 10. Post-filter by normalized threshold to guarantee boundary satisfaction
        if resolved_threshold is not None:
            passages = [
                p
                for p in passages
                if p.relevance_score is not None and p.relevance_score >= resolved_threshold
            ]

        # 11. Guarantee descending relevance score ranking
        passages.sort(
            key=lambda p: (p.relevance_score is not None, p.relevance_score or 0.0),
            reverse=True,
        )

        logger.info("Retrieved %d reference passages", len(passages))
        return passages

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[ReferencePassage]:
        """Convenience alias for retrieve()."""
        return self.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_id=document_id,
            source_type=source_type,
        )

    def retrieve_guidelines(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> List[ReferencePassage]:
        """Convenience method filtering specifically for clinical guidelines."""
        return self.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_id=document_id,
            source_type="clinical_guideline",
        )
