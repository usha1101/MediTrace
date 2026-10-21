"""MediTrace FastAPI Backend Application.

Main entrypoint for the integration backend (Member B).
Serves health check and consultation endpoints with mock responses
until Member A's Clinical AI engine is integrated.
"""

import os
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.config import settings
from backend.schemas.consultation import (
    ConsultationRequest,
    ConsultationResponse,
    HealthResponse,
)
from backend.schemas.transcription import TranscriptionResponse
from backend.services.clinical_ai import (
    ClinicalEngineConfigurationError,
    ClinicalEngineInterface,
    ClinicalWorkflowExecutionError,
    get_clinical_engine,
)
from backend.services.transcription import (
    TranscriptionEngineInterface,
    TranscriptionEngineUnavailableError,
    TranscriptionError,
    get_transcription_engine,
)

app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(ClinicalEngineConfigurationError)
async def clinical_engine_configuration_handler(request, exc: ClinicalEngineConfigurationError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Clinical AI engine configuration error: {str(exc)}"},
    )


@app.exception_handler(ClinicalWorkflowExecutionError)
async def clinical_workflow_execution_handler(request, exc: ClinicalWorkflowExecutionError):
    if exc.stage == "validate_input":
        status_code = status.HTTP_400_BAD_REQUEST
    elif exc.stage in ("extract_clinical_information", "generate_soap_draft"):
        status_code = status.HTTP_502_BAD_GATEWAY
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return JSONResponse(
        status_code=status_code,
        content={"detail": f"Clinical workflow execution error: {str(exc)}"},
    )


# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """Root endpoint providing service overview and documentation links."""
    return {
        "service": "MediTrace Clinical Copilot API",
        "version": settings.version,
        "status": "operational",
        "mock_engine_active": settings.mock_mode,
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
    description="Returns backend service health status, version, and AI engine status.",
)
def health_check() -> HealthResponse:
    """Verify backend health and readiness."""
    return HealthResponse(
        status="healthy",
        service="meditrace-backend",
        version=settings.version,
        mock_engine_active=settings.mock_mode,
    )


@app.post(
    f"{settings.api_prefix}/consultation",
    response_model=ConsultationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Clinical Consultation"],
    summary="Process clinical encounter transcript",
    description=(
        "Accepts ambient consultation transcript and patient metadata, "
        "returning a structured draft SOAP note, evidence references, and clinical safety verification alerts."
    ),
)
def process_consultation(
    request: ConsultationRequest,
    engine: ClinicalEngineInterface = Depends(get_clinical_engine),
) -> ConsultationResponse:
    """Handle consultation note generation and safety verification."""
    try:
        response = engine.generate_consultation(request)
        return response
    except HTTPException:
        raise
    except ClinicalEngineConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Clinical AI engine configuration error: {str(exc)}",
        ) from exc
    except ClinicalWorkflowExecutionError as exc:
        if exc.stage == "validate_input":
            status_code = status.HTTP_400_BAD_REQUEST
        elif exc.stage in ("extract_clinical_information", "generate_soap_draft"):
            status_code = status.HTTP_502_BAD_GATEWAY
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=status_code,
            detail=f"Clinical workflow execution error: {str(exc)}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical engine processing failed: {str(exc)}",
        ) from exc


# Also expose /consultation at root level for flexibility
@app.post(
    "/consultation",
    response_model=ConsultationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Clinical Consultation"],
    include_in_schema=False,
)
def process_consultation_alias(
    request: ConsultationRequest,
    engine: ClinicalEngineInterface = Depends(get_clinical_engine),
) -> ConsultationResponse:
    return process_consultation(request, engine)


@app.post(
    f"{settings.api_prefix}/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Audio Transcription"],
    summary="Transcribe clinical audio encounter",
    description=(
        "Accepts multipart audio upload (.wav, .mp3, .m4a, .ogg, .webm, .flac) "
        "and transcribes speech to text using local faster-whisper or mock engine."
    ),
)
async def transcribe_audio(
    file: UploadFile = File(...),
    engine: TranscriptionEngineInterface = Depends(get_transcription_engine),
) -> TranscriptionResponse:
    """Handle speech-to-text audio upload, validation, and local transcription."""
    # 1. Validate file presence and filename
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file uploaded or filename missing.",
        )

    # 2. Validate file extension
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in settings.allowed_audio_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio format '{ext}'. Supported formats: "
                f"{', '.join(settings.allowed_audio_extensions)}"
            ),
        )

    # 3. Read content and validate size
    try:
        content = await file.read()
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(err)}",
        ) from err

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty (0 bytes).",
        )

    if len(content) > settings.max_audio_size_bytes:
        max_mb = settings.max_audio_size_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Audio file exceeds maximum allowed size of {max_mb} MB.",
        )

    # 4. Process transcription
    try:
        response = engine.transcribe(audio_bytes=content, filename=file.filename)
        return response
    except TranscriptionEngineUnavailableError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Transcription engine unavailable: {str(err)}",
        ) from err
    except TranscriptionError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription processing error: {str(err)}",
        ) from err
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during transcription: {str(err)}",
        ) from err


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Audio Transcription"],
    include_in_schema=False,
)
async def transcribe_audio_alias(
    file: UploadFile = File(...),
    engine: TranscriptionEngineInterface = Depends(get_transcription_engine),
) -> TranscriptionResponse:
    return await transcribe_audio(file, engine)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)

