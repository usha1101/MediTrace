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
]
