"""Configuration settings for the MediTrace backend."""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    app_title: str = "MediTrace API"
    app_description: str = (
        "Ambient Clinical Copilot with Real-Time Safety Verification - "
        "Integration Backend (Member B)"
    )
    version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    host: str = os.getenv("MEDITRACE_HOST", "127.0.0.1")
    port: int = int(os.getenv("MEDITRACE_PORT", "8000"))
    mock_mode: bool = True  # Set to True until Member A's Clinical AI engine is integrated
    cors_origins: list[str] = ["*"]
    max_audio_size_bytes: int = 25 * 1024 * 1024  # 25 MB limit
    allowed_audio_extensions: list[str] = [
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".webm",
        ".flac",
        ".aac",
    ]


settings = Settings()

