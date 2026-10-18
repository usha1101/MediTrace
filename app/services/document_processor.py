"""
MediTrace - Reference Document Processor & Deterministic Chunker
Foundation component for clinical knowledge indexing and RAG pipelines.

CLINICAL SAFETY NOTICE:
-----------------------
MediTrace is an educational clinical documentation prototype.
It must NOT autonomously diagnose, prescribe, or make treatment decisions.
Reference texts processed by this component are for informational retrieval
and clinician-in-the-loop review only. Reference documents must be treated as
passive data, not executable instructions.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

from app.schemas import DocumentChunk

logger = logging.getLogger(__name__)


# =====================================================================
# Document Processing Exceptions
# =====================================================================

class DocumentProcessingError(Exception):
    """Base exception for reference document processing errors."""
    pass


class InvalidDocumentError(DocumentProcessingError, ValueError):
    """Raised when document content, parameters, or metadata are invalid."""
    pass


# =====================================================================
# Document Processor Implementation
# =====================================================================

class DocumentProcessor:
    """
    Deterministic document processor and chunker for clinical reference literature.

    Core capabilities:
    - Normalizes line endings consistently (\\r\\n and \\r to \\n).
    - Preserves paragraph (\\n\\n) and sentence boundaries where practical.
    - Prevents infinite loops through strict monotonic index advancement.
    - Preserves all non-whitespace source content without loss.
    - Generates collision-resistant, deterministic chunk identifiers.
    - Preserves document provenance and metadata across all chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the DocumentProcessor with configurable chunk size and overlap.

        Args:
            chunk_size: Maximum character length for each chunk (must be > 0).
            chunk_overlap: Number of characters shared between adjacent chunks (must be >= 0 and < chunk_size).

        Raises:
            ValueError: If chunk_size or chunk_overlap configuration parameters are invalid.
        """
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be a positive integer greater than 0, got {chunk_size}."
            )
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be a non-negative integer (>= 0), got {chunk_overlap}."
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
            )

        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize line endings and structural whitespace in reference text.

        Args:
            text: Raw input reference text.

        Returns:
            Normalized string with standardized Unix-style (\\n) line endings.
        """
        if not text:
            return ""
        # Standardize CRLF and CR to LF
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return normalized

    def split_text(self, text: str) -> List[str]:
        """
        Split normalized text into deterministic, readable text chunks.

        Prioritizes paragraph boundaries (\\n\\n), then sentence boundaries (. ),
        line breaks (\\n), and whitespace word breaks, falling back to a hard split
        at chunk_size when necessary.

        Strict maximum chunk size:
            No returned chunk will ever exceed `chunk_size` characters.

        Overlap interaction and loop prevention:
            When `chunk_overlap > 0`, the tentative start of the next chunk is
            `split_point - chunk_overlap`.
            To prevent infinite loops and ensure finite forward progress:
                next_start = max(start + 1, split_point - chunk_overlap)
            This guarantees:
                1. Strict progress: `next_start > start`, ensuring the loop always terminates.
                2. Continuous coverage: `next_start <= split_point`, ensuring no non-whitespace
                   text between `start` and `split_point` is silently omitted or lost.

        Args:
            text: Source reference document text.

        Returns:
            List of non-empty text chunk strings, each with length <= chunk_size.

        Raises:
            InvalidDocumentError: If text is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise InvalidDocumentError("Document text must not be empty or whitespace only.")

        normalized = self.normalize_text(text)
        stripped_normalized = normalized.strip()

        # If the entire text fits within a single chunk budget, return it directly
        if len(stripped_normalized) <= self.chunk_size:
            return [stripped_normalized]

        chunks: List[str] = []
        start = 0
        text_len = len(normalized)

        while start < text_len:
            # Skip leading empty newlines at chunk start boundary if not the very beginning
            while start < text_len and normalized[start] in "\r\n":
                start += 1
            if start >= text_len:
                break

            end = start + self.chunk_size

            if end >= text_len:
                # Tail segment
                chunk_str = normalized[start:text_len].strip()
                if chunk_str:
                    # Strict chunk size enforcement
                    if len(chunk_str) > self.chunk_size:
                        chunk_str = chunk_str[: self.chunk_size].rstrip()
                    chunks.append(chunk_str)
                break

            # Search for natural boundaries
            # 1. First priority: look for paragraph break (\n\n) in [start + chunk_size // 3, end]
            p_search_start = start + max(1, self.chunk_size // 3)
            p_segment = normalized[p_search_start:end]
            p_idx = p_segment.rfind("\n\n")

            split_offset = -1
            if p_idx != -1:
                split_offset = p_search_start + p_idx + 2
            else:
                # 2. Fall back to sentence/line/word boundary in [end - search_window, end]
                search_window = max(self.chunk_overlap, self.chunk_size // 3)
                search_start = max(start + 1, end - search_window)
                segment = normalized[search_start:end]

                # Sentence terminal punctuation followed by space or newline
                sentence_matches = list(re.finditer(r"[.?!][\s\n]", segment))
                if sentence_matches:
                    split_offset = search_start + sentence_matches[-1].end()
                else:
                    # Single newline
                    n_idx = segment.rfind("\n")
                    if n_idx != -1:
                        split_offset = search_start + n_idx + 1
                    else:
                        # Space boundary
                        s_idx = segment.rfind(" ")
                        if s_idx != -1:
                            split_offset = search_start + s_idx + 1

            # Fall back to hard end boundary if no suitable break point found
            if split_offset == -1 or split_offset <= start:
                split_point = end
            else:
                split_point = split_offset

            # Strict upper bound: split_point must never exceed end (start + chunk_size)
            split_point = min(split_point, end)

            chunk_str = normalized[start:split_point].strip()
            # Enforce strict maximum chunk size
            if len(chunk_str) > self.chunk_size:
                chunk_str = chunk_str[: self.chunk_size].rstrip()

            if chunk_str:
                chunks.append(chunk_str)

            # Advance next start with overlap.
            # Invariant: start < next_start <= split_point
            tentative_start = max(start + 1, split_point - self.chunk_overlap)

            # Snap tentative_start to a natural word boundary if it lands inside a word
            # and a whitespace boundary exists before split_point
            if (
                self.chunk_overlap > 0
                and tentative_start < split_point
                and tentative_start > 0
                and not normalized[tentative_start - 1].isspace()
            ):
                ws_match = re.search(r"\s+", normalized[tentative_start:split_point])
                if ws_match:
                    aligned_start = tentative_start + ws_match.end()
                    if aligned_start < split_point:
                        tentative_start = aligned_start

            next_start = max(start + 1, tentative_start)
            start = next_start


        return chunks

    @staticmethod
    def generate_chunk_id(
        document_id: str,
        chunk_index: int,
        chunk_text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> str:
        """
        Generate a deterministic, collision-resistant identifier for a chunk.

        Combines document_id, chunk_index, chunking settings, and text content
        using SHA-256 to ensure identical configurations and text always produce the identical ID.

        Args:
            document_id: Identifier of the parent document.
            chunk_index: Sequential 0-based chunk index.
            chunk_text: Exact verbatim chunk text content.
            chunk_size: Optional configured chunk size.
            chunk_overlap: Optional configured chunk overlap.

        Returns:
            Deterministic string identifier.
        """
        sanitized_doc_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", document_id).strip("_")
        settings_str = f":size={chunk_size}:overlap={chunk_overlap}" if chunk_size is not None else ""
        payload = f"{document_id}:{chunk_index}{settings_str}:{chunk_text}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{sanitized_doc_id}_chunk_{chunk_index:04d}_{digest}"

    def process_document(
        self,
        text: str,
        document_id: str,
        source_title: str,
        source_type: str,
        section: Optional[str] = None,
        publication_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Process a reference document text into a validated sequence of DocumentChunk models.

        Args:
            text: Verbatim reference document text.
            document_id: Unique identifier for the parent document.
            source_title: Title of the source guideline, article, or monograph.
            source_type: Classification (e.g. 'clinical_guideline', 'drug_label', 'protocol').
            section: Optional section heading if known (never fabricated).
            publication_date: Optional publication or revision date if known (never fabricated).
            metadata: Optional dictionary of additional provenance metadata.

        Returns:
            List of validated DocumentChunk instances.

        Raises:
            InvalidDocumentError: If text, document_id, source_title, or source_type are invalid.
        """
        # Validate required source metadata
        if not document_id or not document_id.strip():
            raise InvalidDocumentError("document_id must not be empty or whitespace only.")
        if not source_title or not source_title.strip():
            raise InvalidDocumentError("source_title must not be empty or whitespace only.")
        if not source_type or not source_type.strip():
            raise InvalidDocumentError("source_type must not be empty or whitespace only.")

        clean_doc_id = document_id.strip()
        clean_title = source_title.strip()
        clean_type = source_type.strip()
        clean_section = section.strip() if section and section.strip() else None
        clean_pub_date = publication_date.strip() if publication_date and publication_date.strip() else None
        base_metadata = dict(metadata) if metadata else {}

        # Split text into chunks
        text_chunks = self.split_text(text)
        total_chunks = len(text_chunks)

        processed_chunks: List[DocumentChunk] = []

        for idx, chunk_text in enumerate(text_chunks):
            # Strict invariant check: chunk length must never exceed configured chunk_size
            if len(chunk_text) > self.chunk_size:
                raise DocumentProcessingError(
                    f"Chunk {idx} length ({len(chunk_text)}) exceeded strict maximum chunk_size ({self.chunk_size})."
                )

            passage_id = self.generate_chunk_id(
                clean_doc_id,
                idx,
                chunk_text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            chunk_meta = {
                **base_metadata,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_length_chars": len(chunk_text),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }

            chunk_model = DocumentChunk(
                passage_id=passage_id,
                document_id=clean_doc_id,
                text=chunk_text,
                source_title=clean_title,
                source_type=clean_type,
                section=clean_section,
                publication_date=clean_pub_date,
                chunk_index=idx,
                metadata=chunk_meta,
            )
            processed_chunks.append(chunk_model)

        logger.info(
            "Processed document '%s' into %d chunks (chunk_size=%d, overlap=%d)",
            clean_doc_id,
            total_chunks,
            self.chunk_size,
            self.chunk_overlap,
        )

        return processed_chunks

