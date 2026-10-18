"""
MediTrace application services package.
"""

from app.services.clinical_extraction import (
    CLINICAL_EXTRACTION_SYSTEM_INSTRUCTION,
    ClinicalExtractionError,
    ClinicalExtractionService,
    ExtractionServiceError,
    InvalidTranscriptError,
)
from app.services.llm_service import (
    DEFAULT_GEMINI_MODEL,
    GeminiLLMService,
    LLMAPIError,
    LLMConfigurationError,
    LLMEmptyResponseError,
    LLMResponseParsingError,
    LLMServiceError,
    LLMValidationError,
)

from app.services.document_processor import (
    DocumentProcessingError,
    DocumentProcessor,
    InvalidDocumentError,
)
from app.services.embedding_service import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    ChunkEmbedding,
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    EmbeddingService,
    EmbeddingServiceError,
    EmbeddingValidationError,
)
from app.services.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_STORAGE_PATH,
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

__all__ = [
    # LLM Service
    "DEFAULT_GEMINI_MODEL",
    "GeminiLLMService",
    "LLMServiceError",
    "LLMConfigurationError",
    "LLMAPIError",
    "LLMEmptyResponseError",
    "LLMResponseParsingError",
    "LLMValidationError",
    # Clinical Extraction
    "ClinicalExtractionService",
    "ClinicalExtractionError",
    "InvalidTranscriptError",
    "ExtractionServiceError",
    "CLINICAL_EXTRACTION_SYSTEM_INSTRUCTION",
    # Document Processing & Chunker
    "DocumentProcessor",
    "DocumentProcessingError",
    "InvalidDocumentError",
    # Embedding Service
    "EmbeddingService",
    "ChunkEmbedding",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSION",
    "EmbeddingServiceError",
    "EmbeddingConfigurationError",
    "EmbeddingAPIError",
    "EmbeddingValidationError",
    # Vector Store Service
    "VectorStore",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_STORAGE_PATH",
    "DEFAULT_VECTOR_DIMENSION",
    "VectorStoreError",
    "VectorStoreConfigurationError",
    "VectorStoreValidationError",
    "VectorStoreInsertionError",
    "VectorStoreSearchError",
    "cosine_similarity_to_relevance_score",
    "passage_id_to_point_id",
]
