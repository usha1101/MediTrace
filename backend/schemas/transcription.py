"""Pydantic schemas for audio transcription endpoint."""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Response payload for audio speech-to-text transcription."""
    filename: str = Field(..., description="Name of the uploaded audio file")
    transcript: str = Field(..., description="Transcribed text from audio")
    language: Optional[str] = Field(default="en", description="Detected or configured spoken language")
    duration_seconds: Optional[float] = Field(default=None, description="Audio duration in seconds")
    engine_name: str = Field(..., description="Name of the transcription engine used")
    is_mock: bool = Field(default=False, description="Flag indicating if mock transcription was used")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp when transcription was generated"
    )
