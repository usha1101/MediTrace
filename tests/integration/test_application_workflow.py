"""
Integration Test Suite for the MediTrace LangGraph Clinical Workflow.
Tests the end-to-end StateGraph pipeline connecting Member A's Clinical AI services.

RATIONALE & SAFETY:
-------------------
All external API boundaries (Gemini LLM and Embedding endpoints) are mocked using deterministic,
offline mocks. Internal business logic, schema validations, in-memory Qdrant storage,
deterministic safety verification rules, and state machine routing are executed using REAL services.
All tests run 100% offline with zero live network calls, zero API keys required, and purely synthetic clinical data.
"""

import os
from unittest.mock import MagicMock
import pytest

from app.schemas import (
    CLINICAL_SAFETY_DISCLAIMER,
    AllergyInfo,
    CertaintyLevel,
    ClinicalExtractionResult,
    DocumentChunk,
    DraftSOAPNote,
    InformationSource,
    MedicationMention,
    MedicationStatus,
    MissingInformation,
    NoteStatus,
    ObservationInfo,
    PresenceStatus,
    ReferencePassage,
    SafetyCategory,
    SafetyReviewFlag,
    SafetySeverity,
    SymptomInfo,
    VitalSigns,
)
from app.services.clinical_extraction import (
    ClinicalExtractionService,
    ExtractionServiceError,
)
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import GeminiLLMService
from app.services.retrieval_service import (
    RetrievalService,
    RetrievalServiceError,
)
from app.services.safety_verification import (
    ClinicalSafetyVerificationService,
    SafetyVerificationError,
)
from app.services.soap_note_service import (
    SOAPNoteGenerationError,
    SOAPNoteService,
)
from app.services.vector_store import VectorStore

from backend.workflows.clinical_graph import (
    ClinicalState,
    ClinicalWorkflow,
    ClinicalWorkflowServices,
    create_clinical_graph,
)


# =====================================================================
# Synthetic Clinical Fixtures & Vectors
# =====================================================================

SYNTHETIC_TRANSCRIPT: str = (
    "Doctor: Good morning, Mrs. Davis. What brings you in today?\n"
    "Patient: I developed an itchy skin rash and hives across my arms and chest yesterday.\n"
    "Doctor: Are you having shortness of breath, lip swelling, or trouble swallowing?\n"
    "Patient: No lip swelling and no shortness of breath at all.\n"
    "Doctor: Any new medications recently?\n"
    "Patient: Yes, yesterday my dentist prescribed Amoxicillin 500mg three times daily for a tooth infection.\n"
    "Doctor: Do you have any known drug allergies?\n"
    "Patient: Yes, I had severe hives and a rash from Penicillin ten years ago.\n"
    "Doctor: Let us check your vitals. Blood pressure 120/80 mmHg, heart rate 72 bpm, oxygen saturation 98%.\n"
    "Doctor: We must stop the Amoxicillin immediately because of high cross-reactivity with your penicillin allergy."
)

SYNTHETIC_GUIDELINE_TEXT: str = (
    "CLINICAL GUIDELINE: Beta-Lactam Allergy and Cross-Reactivity Management.\n"
    "Document ID: GUIDELINE-BL-2024.\n"
    "Patients with confirmed IgE-mediated penicillin allergy are at high risk of hypersensitivity "
    "when exposed to aminopenicillins including amoxicillin and ampicillin. Co-amoxiclav carries identical risk. "
    "Amoxicillin is contraindicated in patients with documented penicillin allergy. "
    "Immediately discontinue therapy and document allergy profile."
)


def make_deterministic_vector(seed_value: float) -> list[float]:
    """Generate a deterministic, normalized 768-dim vector for offline testing."""
    vec = [0.0] * 768
    vec[0] = float(seed_value)
    for i in range(1, 768):
        vec[i] = round(((i * 0.001) % 0.01), 4)
    return vec


@pytest.fixture
def synthetic_extraction() -> ClinicalExtractionResult:
    """Deterministic synthetic extraction result for Mrs. Davis encounter."""
    return ClinicalExtractionResult(
        encounter_id="ENC-DAVIS-999",
        symptoms=[
            SymptomInfo(
                name="Itchy skin rash",
                status=PresenceStatus.PRESENT,
                certainty=CertaintyLevel.UNCERTAIN,
                source=InformationSource.PATIENT_REPORTED,
                duration="1 day",
            ),
            SymptomInfo(
                name="Hives",
                status=PresenceStatus.PRESENT,
                certainty=CertaintyLevel.UNCERTAIN,
                source=InformationSource.PATIENT_REPORTED,
            ),
            SymptomInfo(
                name="Shortness of breath",
                status=PresenceStatus.ABSENT,
                certainty=CertaintyLevel.DENIED,
                source=InformationSource.PATIENT_REPORTED,
            ),
        ],
        allergies=[
            AllergyInfo(
                allergen="Penicillin",
                reaction="Severe hives and rash",
                reaction_severity="severe",
                source=InformationSource.PATIENT_REPORTED,
                verified_by_clinician=False,
            )
        ],
        medications=[
            MedicationMention(
                name="Amoxicillin",
                dosage="500 mg",
                frequency="three times daily",
                route="oral",
                status=MedicationStatus.CURRENT,
            )
        ],
        vitals=VitalSigns(
            blood_pressure_systolic=120,
            blood_pressure_diastolic=80,
            heart_rate=72,
            oxygen_saturation_pct=98.0,
        ),
        observations=[
            ObservationInfo(
                category="physical_exam",
                description="Urticarial rash observed across upper extremities.",
                is_normal=False,
            )
        ],
        missing_information=[
            MissingInformation(
                category="allergy_history",
                description="Penicillin allergy was unasked prior to dental prescribing.",
                clinical_relevance="Pre-prescription allergy screening is essential.",
            )
        ],
        consultation_summary="Patient presented with acute hives after Amoxicillin in context of penicillin allergy.",
    )


@pytest.fixture
def synthetic_draft_soap() -> DraftSOAPNote:
    """Deterministic synthetic draft SOAP note."""
    return DraftSOAPNote(
        note_id="soap_ENC-DAVIS-999",
        subjective="Patient reports acute itchy hives after starting Amoxicillin. Denies shortness of breath.",
        objective="Vitals: BP 120/80 mmHg, HR 72 bpm, SpO2 98%. Exam: Urticarial rash on upper extremities.",
        assessment="Provisional drug-induced urticaria secondary to Amoxicillin with known penicillin allergy.",
        plan="1. Discontinue Amoxicillin immediately.\n2. Supportive antihistamine therapy.\n3. Clinician to select alternative.",
        status=NoteStatus.DRAFT_PENDING_REVIEW,
        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
    )


@pytest.fixture
def in_memory_retrieval_service() -> RetrievalService:
    """Provides a real RetrievalService backed by in-memory Qdrant populated with synthetic guideline chunks."""
    doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    chunks = doc_processor.process_document(
        text=SYNTHETIC_GUIDELINE_TEXT,
        document_id="DOC-BETA-LACTAM-01",
        source_title="Beta-Lactam Allergy Management",
        source_type="clinical_guideline",
    )

    guideline_vector = make_deterministic_vector(1.0)
    query_vector = make_deterministic_vector(0.98)

    mock_client = MagicMock()

    def mock_embed(*args, **kwargs):
        config = kwargs.get("config")
        task_type = config.task_type if config else None
        vec = query_vector if task_type == "RETRIEVAL_QUERY" else guideline_vector
        mock_e = MagicMock()
        mock_e.values = vec
        resp = MagicMock()
        resp.embeddings = [mock_e]
        return resp

    mock_client.models.embed_content.side_effect = mock_embed

    embedding_service = EmbeddingService(
        api_key="synthetic-test-key",
        output_dimensionality=768,
        client=mock_client,
    )
    chunk_embeddings = embedding_service.embed_chunks(chunks)

    vector_store = VectorStore(
        collection_name="test_clinical_guidelines",
        storage_path=":memory:",
        vector_dim=768,
    )
    vector_store.upsert_embeddings(chunk_embeddings)

    return RetrievalService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        default_top_k=2,
    )


# =====================================================================
# Workflow Test Cases (15 Scenarios)
# =====================================================================

def test_successful_end_to_end_workflow(
    synthetic_extraction, synthetic_draft_soap, in_memory_retrieval_service
):
    """Scenario 1: Verify successful end-to-end execution of the complete 7-stage workflow."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    safety_svc = ClinicalSafetyVerificationService()

    mock_llm_soap = MagicMock(spec=GeminiLLMService)
    mock_llm_soap.generate_structured.return_value = synthetic_draft_soap
    soap_svc = SOAPNoteService(llm_service=mock_llm_soap)

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=in_memory_retrieval_service,
        safety_service=safety_svc,
        soap_service=soap_svc,
    )

    workflow = ClinicalWorkflow(services=services)
    result = workflow.invoke({
        "encounter_id": "ENC-DAVIS-999",
        "transcript": SYNTHETIC_TRANSCRIPT,
        "known_allergies": ["Penicillin"],
    })

    # Assert workflow completed successfully
    assert result["workflow_status"] == "completed"
    assert result["error"] is None
    assert result["error_stage"] is None

    # Assert structured extraction output
    assert result["extraction_result"] is not None
    assert len(result["extraction_result"].symptoms) == 3

    # Assert query formulation
    assert result["retrieval_query"] is not None
    assert "Amoxicillin" in result["retrieval_query"]
    assert "Penicillin" in result["retrieval_query"]

    # Assert retrieved references
    assert result["retrieval_status"] == "success"
    assert len(result["reference_passages"]) >= 1
    assert result["reference_passages"][0].source_document == "Beta-Lactam Allergy Management"

    # Assert safety verification flags
    assert len(result["safety_flags"]) >= 1
    cross_react_flags = [
        f for f in result["safety_flags"]
        if f.category == SafetyCategory.DRUG_ALLERGY_CONFLICT and "Amoxicillin" in f.flagged_items
    ]
    assert len(cross_react_flags) >= 1
    assert cross_react_flags[0].severity in (SafetySeverity.HIGH, SafetySeverity.CRITICAL)

    # Assert draft SOAP note
    note = result["draft_soap_note"]
    assert note is not None
    assert note.status == NoteStatus.DRAFT_PENDING_REVIEW
    assert CLINICAL_SAFETY_DISCLAIMER in note.disclaimer
    assert len(note.safety_flags) >= 1
    assert len(note.references) >= 1


def test_empty_consultation_text_fails_gracefully():
    """Scenario 2: Verify validation failure halts workflow when transcript is empty or too short."""
    workflow = ClinicalWorkflow()

    result = workflow.invoke({
        "encounter_id": "ENC-EMPTY-01",
        "transcript": "   Too short   ",  # under 10 chars after strip
    })

    assert result["workflow_status"] == "failed"
    assert result["error_stage"] == "validate_input"
    assert "transcript must be a non-empty string" in result["error"].lower()
    assert result["draft_soap_note"] is None
    assert result["extraction_result"] is None


def test_clinical_extraction_failure_halts_workflow(in_memory_retrieval_service):
    """Scenario 3: Verify workflow halts safely when ClinicalExtractionService fails."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.side_effect = ExtractionServiceError("LLM response malformed")
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=in_memory_retrieval_service,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-EXTRACT-FAIL",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert result["workflow_status"] == "failed"
    assert result["error_stage"] == "extract_clinical_information"
    assert "Clinical extraction failed" in result["error"]
    assert result["draft_soap_note"] is None
    assert result["retrieval_query"] is None


def test_retrieval_returning_no_references(synthetic_extraction, synthetic_draft_soap):
    """Scenario 4: Verify workflow continues safely when retrieval finds zero matches without fabricating citations."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    # Empty retrieval service mock returning 0 passages
    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    mock_llm_soap = MagicMock(spec=GeminiLLMService)
    # Return SOAP note with empty references
    soap_note_empty_refs = synthetic_draft_soap.model_copy(update={"references": []})
    mock_llm_soap.generate_structured.return_value = soap_note_empty_refs
    soap_svc = SOAPNoteService(llm_service=mock_llm_soap)

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
        soap_service=soap_svc,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-NO-REFS",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert result["workflow_status"] == "completed"
    assert result["retrieval_status"] == "no_evidence_found"
    assert len(result["reference_passages"]) == 0
    assert result["draft_soap_note"] is not None
    assert len(result["draft_soap_note"].references) == 0


def test_retrieval_failure_handled_gracefully(synthetic_extraction, synthetic_draft_soap):
    """Scenario 5: Verify retrieval exceptions mark status as unavailable without crashing downstream note."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.side_effect = RetrievalServiceError("Vector database connection timed out")

    mock_llm_soap = MagicMock(spec=GeminiLLMService)
    mock_llm_soap.generate_structured.return_value = synthetic_draft_soap.model_copy(update={"references": []})
    soap_svc = SOAPNoteService(llm_service=mock_llm_soap)

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
        soap_service=soap_svc,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-RETRIEVAL-ERR",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert result["workflow_status"] == "completed"
    assert result["retrieval_status"] == "unavailable_error"
    assert len(result["reference_passages"]) == 0
    assert result["draft_soap_note"] is not None


def test_safety_verification_returning_flags(synthetic_extraction, synthetic_draft_soap):
    """Scenario 6: Verify deterministic safety flags (penicillin allergy conflict) are detected and stored."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_llm_soap = MagicMock(spec=GeminiLLMService)
    mock_llm_soap.generate_structured.return_value = synthetic_draft_soap
    soap_svc = SOAPNoteService(llm_service=mock_llm_soap)

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
        soap_service=soap_svc,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-ALLERGY-CHECK",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert result["workflow_status"] == "completed"
    flags = result["safety_flags"]
    assert len(flags) >= 1
    conflict_flag = next(
        (f for f in flags if f.category == SafetyCategory.DRUG_ALLERGY_CONFLICT and "Amoxicillin" in f.flagged_items),
        None,
    )
    assert conflict_flag is not None
    assert "Amoxicillin" in conflict_flag.flagged_items
    assert "Penicillin" in conflict_flag.flagged_items


def test_safety_verification_failure_halts_workflow(synthetic_extraction):
    """Scenario 7: Verify safety verification unexpected error halts workflow and prevents SOAP synthesis."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_safety = MagicMock(spec=ClinicalSafetyVerificationService)
    mock_safety.verify.side_effect = SafetyVerificationError("Safety engine validation crashed")

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
        safety_service=mock_safety,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-SAFETY-FAIL",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert result["workflow_status"] == "failed"
    assert result["error_stage"] == "verify_safety"
    assert "Safety verification failed" in result["error"]
    assert result["draft_soap_note"] is None


def test_soap_generation_failure_returns_failed_workflow(synthetic_extraction):
    """Scenario 8: Verify SOAP synthesis error results in explicit failure without fabricating notes."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_soap = MagicMock(spec=SOAPNoteService)
    mock_soap.generate_soap_note.side_effect = SOAPNoteGenerationError("LLM response schema validation failed")

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
        soap_service=mock_soap,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-SOAP-FAIL",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert result["workflow_status"] == "failed"
    assert result["error_stage"] == "generate_soap_draft"
    assert "SOAP note generation failed" in result["error"]
    assert result["draft_soap_note"] is None


def test_missing_information_preservation(synthetic_extraction, synthetic_draft_soap):
    """Scenario 9: Verify missing clinical information identified during extraction is preserved."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_llm_soap = MagicMock(spec=GeminiLLMService)
    mock_llm_soap.generate_structured.return_value = synthetic_draft_soap.model_copy(
        update={"missing_information": synthetic_extraction.missing_information}
    )
    soap_svc = SOAPNoteService(llm_service=mock_llm_soap)

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
        soap_service=soap_svc,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-MISSING-INFO",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert len(result["missing_information"]) == 1
    assert result["missing_information"][0].category == "allergy_history"
    assert result["draft_soap_note"] is not None
    assert len(result["draft_soap_note"].missing_information) == 1


def test_reference_provenance_preservation(
    synthetic_extraction, synthetic_draft_soap, in_memory_retrieval_service
):
    """Scenario 10: Verify reference provenance metadata (source document, score) is attached to draft note."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_llm_soap = MagicMock(spec=GeminiLLMService)
    mock_llm_soap.generate_structured.return_value = synthetic_draft_soap
    soap_svc = SOAPNoteService(llm_service=mock_llm_soap)

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=in_memory_retrieval_service,
        soap_service=soap_svc,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-PROVENANCE",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    passages = result["reference_passages"]
    assert len(passages) >= 1
    top_p = passages[0]
    assert isinstance(top_p, ReferencePassage)
    assert top_p.source_document == "Beta-Lactam Allergy Management"
    assert top_p.relevance_score is not None
    assert top_p.relevance_score >= 0.70
    assert result["draft_soap_note"] is not None


def test_no_fabricated_references(synthetic_extraction, synthetic_draft_soap):
    """Scenario 11: Verify zero references are attached when retrieval returns nothing (no hallucinations)."""
    mock_llm_extract = MagicMock(spec=GeminiLLMService)
    mock_llm_extract.generate_structured.return_value = synthetic_extraction
    extraction_svc = ClinicalExtractionService(llm_service=mock_llm_extract)

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    mock_llm_soap = MagicMock(spec=GeminiLLMService)
    mock_llm_soap.generate_structured.return_value = synthetic_draft_soap.model_copy(update={"references": []})
    soap_svc = SOAPNoteService(llm_service=mock_llm_soap)

    services = ClinicalWorkflowServices(
        extraction_service=extraction_svc,
        retrieval_service=mock_retrieval,
        soap_service=soap_svc,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-NO-HALLUCINATED-REFS",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert len(result["reference_passages"]) == 0
    assert len(result["draft_soap_note"].references) == 0


def test_no_gemini_network_calls_during_tests(monkeypatch, synthetic_extraction, synthetic_draft_soap):
    """Scenario 12: Verify that no live external network requests are made during unit testing."""
    # Poison socket connection to guarantee zero external network access
    import socket

    def guarded_connect(*args, **kwargs):
        raise AssertionError("Network call attempted during offline workflow test!")

    monkeypatch.setattr(socket, "create_connection", guarded_connect)

    mock_llm = MagicMock(spec=GeminiLLMService)
    mock_llm.generate_structured.return_value = synthetic_extraction

    mock_soap_llm = MagicMock(spec=GeminiLLMService)
    mock_soap_llm.generate_structured.return_value = synthetic_draft_soap

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    services = ClinicalWorkflowServices(
        extraction_service=ClinicalExtractionService(llm_service=mock_llm),
        retrieval_service=mock_retrieval,
        soap_service=SOAPNoteService(llm_service=mock_soap_llm),
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-OFFLINE-01",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert result["workflow_status"] == "completed"
    assert result["draft_soap_note"] is not None


def test_no_api_key_required_in_tests(monkeypatch, synthetic_extraction, synthetic_draft_soap):
    """Scenario 13: Verify workflow operates with injected services even when GEMINI_API_KEY is not set."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert os.getenv("GEMINI_API_KEY") is None

    mock_llm = MagicMock(spec=GeminiLLMService)
    mock_llm.generate_structured.return_value = synthetic_extraction

    mock_soap_llm = MagicMock(spec=GeminiLLMService)
    mock_soap_llm.generate_structured.return_value = synthetic_draft_soap

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    services = ClinicalWorkflowServices(
        extraction_service=ClinicalExtractionService(llm_service=mock_llm),
        retrieval_service=mock_retrieval,
        soap_service=SOAPNoteService(llm_service=mock_soap_llm),
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-NO-API-KEY",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    assert result["workflow_status"] == "completed"
    assert result["error"] is None


def test_draft_status_and_disclaimer_preservation(synthetic_extraction, synthetic_draft_soap):
    """Scenario 14: Verify output note strictly preserves DRAFT status and clinical safety disclaimer."""
    mock_llm = MagicMock(spec=GeminiLLMService)
    mock_llm.generate_structured.return_value = synthetic_extraction

    mock_soap_llm = MagicMock(spec=GeminiLLMService)
    mock_soap_llm.generate_structured.return_value = synthetic_draft_soap

    mock_retrieval = MagicMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = []

    services = ClinicalWorkflowServices(
        extraction_service=ClinicalExtractionService(llm_service=mock_llm),
        retrieval_service=mock_retrieval,
        soap_service=SOAPNoteService(llm_service=mock_soap_llm),
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-DISCLAIMER-CHECK",
        "transcript": SYNTHETIC_TRANSCRIPT,
    })

    note = result["draft_soap_note"]
    assert note is not None
    assert note.status == NoteStatus.DRAFT_PENDING_REVIEW
    assert note.disclaimer == CLINICAL_SAFETY_DISCLAIMER
    assert "EDUCATIONAL PROTOTYPE ONLY" in note.disclaimer


def test_conditional_routing_after_failures():
    """Scenario 15: Verify conditional routing skips downstream generation when validation fails."""
    mock_ext = MagicMock(spec=ClinicalExtractionService)
    mock_soap = MagicMock(spec=SOAPNoteService)

    services = ClinicalWorkflowServices(
        extraction_service=mock_ext,
        soap_service=mock_soap,
    )
    workflow = ClinicalWorkflow(services=services)

    result = workflow.invoke({
        "encounter_id": "ENC-FAIL-ROUTE",
        "transcript": "",  # Empty transcript triggers validation error
    })

    assert result["workflow_status"] == "failed"
    assert result["error_stage"] == "validate_input"

    # Verify downstream services were NEVER invoked
    mock_ext.extract_clinical_information.assert_not_called()
    mock_soap.generate_soap_note.assert_not_called()
