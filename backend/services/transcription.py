"""Modular Speech-to-Text (STT) Transcription Service.

Supports:
1. Local execution using `faster-whisper` (on-device, private, no cloud transmission).
2. Mock transcription engine for automated testing and environments without model weights.
"""

from abc import ABC, abstractmethod
import os
import re
import tempfile
from typing import Optional
from backend.schemas.transcription import TranscriptionResponse


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""
    pass


class TranscriptionEngineUnavailableError(TranscriptionError):
    """Raised when the requested transcription engine cannot be loaded."""
    pass


class TranscriptionEngineInterface(ABC):
    """Abstract interface for speech-to-text transcription engines."""

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str) -> TranscriptionResponse:
        """Transcribe audio bytes into text.

        Args:
            audio_bytes: Raw bytes of the audio file.
            filename: Original filename (used to determine format/extension).

        Returns:
            TranscriptionResponse containing transcribed text and metadata.
        """
        pass


class MockTranscriptionEngine(TranscriptionEngineInterface):
    """Deterministic mock transcription engine for testing and environments without model weights."""

    def __init__(self, simulated_transcript: Optional[str] = None):
        self.simulated_transcript = simulated_transcript or (
            "Doctor: Good morning, what brings you in today? "
            "Patient: I've had a severe sore throat, body aches, and fever for the past two days. "
            "Doctor: Let me examine your pharynx. There is marked erythema and tonsillar exudate. "
            "We should consider starting an appropriate antibiotic."
        )

    def transcribe(self, audio_bytes: bytes, filename: str) -> TranscriptionResponse:
        if not audio_bytes:
            raise TranscriptionError("Cannot transcribe empty audio buffer.")

        return TranscriptionResponse(
            filename=filename,
            transcript=self.simulated_transcript,
            language="en",
            duration_seconds=14.5,
            engine_name="MOCK_TRANSCRIPTION_ENGINE (Testing Scaffold)",
            is_mock=True,
        )


class FasterWhisperTranscriptionEngine(TranscriptionEngineInterface):
    """Local, private on-device transcription engine powered by faster-whisper.

    Privacy & Data Flow:
    - Runs 100% locally on CPU or GPU.
    - Zero audio data or transcripts are ever transmitted to any external server.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size or os.getenv("MEDITRACE_WHISPER_MODEL", "base")
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        """Lazy load the Whisper model on first invocation."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except ImportError as err:
                raise TranscriptionEngineUnavailableError(
                    "faster-whisper is not installed. Install it via `pip install faster-whisper` "
                    "or set MEDITRACE_STT_ENGINE=mock."
                ) from err
            except Exception as err:
                raise TranscriptionEngineUnavailableError(
                    f"Failed to load Whisper model '{self.model_size}': {str(err)}. "
                    "Ensure internet connection for initial model download or set MEDITRACE_STT_ENGINE=mock."
                ) from err
        return self._model

    def transcribe(self, audio_bytes: bytes, filename: str) -> TranscriptionResponse:
        if not audio_bytes:
            raise TranscriptionError("Cannot transcribe empty audio buffer.")

        model = self._get_model()

        # Determine file suffix
        _, ext = os.path.splitext(filename)
        suffix = ext if ext else ".wav"

        # Write to temporary file for audio decoding by faster-whisper (av/ffmpeg)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_file.close()

            segments, info = model.transcribe(
                temp_file.name,
                beam_size=5,
                language="en",
                vad_filter=True,
            )

            # Assemble transcribed text
            segment_list = list(segments)
            transcript = " ".join(s.text.strip() for s in segment_list).strip()

            # Format explicit Doctor/Patient labels when followed by a comma.
            transcript = re.sub(
                r"(?i)(^|(?<=[.!?])\s+)(Doctor|Patient),\s*",
                lambda match: f"{match.group(1)}{match.group(2)}: ",
                transcript,
            )

            if not transcript:
                transcript = "[No intelligible speech detected in audio encounter]"

            return TranscriptionResponse(
                filename=filename,
                transcript=transcript,
                language=info.language or "en",
                duration_seconds=round(info.duration, 2) if info.duration else None,
                engine_name=f"faster-whisper (local-{self.model_size})",
                is_mock=False,
            )
        except Exception as err:
            raise TranscriptionError(f"Audio transcription failed: {str(err)}") from err
        finally:
            if os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except OSError:
                    pass


# Singleton engine instances
_faster_whisper_engine: Optional[FasterWhisperTranscriptionEngine] = None
_mock_engine = MockTranscriptionEngine()


def get_transcription_engine() -> TranscriptionEngineInterface:
    """Dependency provider for FastAPI transcription endpoint.

    Reads `MEDITRACE_STT_ENGINE` environment variable:
    - If set to 'mock': returns MockTranscriptionEngine.
    - Otherwise: returns FasterWhisperTranscriptionEngine (lazy loaded).
    """
    global _faster_whisper_engine
    engine_choice = os.getenv("MEDITRACE_STT_ENGINE", "faster-whisper").lower().strip()

    if engine_choice == "mock":
        return _mock_engine

    if _faster_whisper_engine is None:
        _faster_whisper_engine = FasterWhisperTranscriptionEngine()

    return _faster_whisper_engine
