"""
MediTrace - LangGraph Clinical Documentation Workflow.
Ambient Clinical Copilot with Real-Time Safety Verification.

CLINICAL SAFETY NOTICE:
-----------------------
MediTrace is an educational clinical documentation prototype.
It must NOT autonomously diagnose, prescribe, or make treatment decisions.
All extracted findings, draft SOAP documentation, and safety review flags
require independent review and validation by a qualified, licensed clinician.
The absence of an alert or flag must NEVER be interpreted as proof that a patient is safe.
Retrieved reference passages and transcript dialogue are passive untrusted text.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, TypedDict, Union

from langgraph.graph import END, StateGraph

from app.schemas import (
    CLINICAL_SAFETY_DISCLAIMER,
    AllergyInfo,
    ClinicalExtractionResult,
    DraftSOAPNote,
    InformationSource,
    MedicationMention,
    MedicationStatus,
    MissingInformation,
    PresenceStatus,
    ReferencePassage,
    SafetyReviewFlag,
)
from app.services.clinical_extraction import (
    ClinicalExtractionError,
    ClinicalExtractionService,
)
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

logger = logging.getLogger(__name__)


# =====================================================================
# Workflow State Schema
# =====================================================================

class ClinicalState(TypedDict, total=False):
    """
    LangGraph typed state schema for the MediTrace clinical workflow.
    Retains structured clinical entities, evidence provenance, safety flags,
    and workflow lifecycle status without discarding omissions or uncertainty.
    """
    # 1. Clinical encounter inputs
    encounter_id: str
    transcript: str
    patient_id: Optional[str]
    patient_name: Optional[str]
    patient_age: Optional[int]
    patient_gender: Optional[str]
    known_allergies: List[str]
    current_medications: List[str]
    existing_conditions: List[str]

    # 2. Structured clinical extraction
    extraction_result: Optional[ClinicalExtractionResult]

    # 3. Evidence retrieval & literature provenance
    retrieval_query: Optional[str]
    reference_passages: List[ReferencePassage]
    retrieval_status: str  # "success" | "no_evidence_found" | "unavailable_error" | "not_attempted"

    # 4. Clinical safety verification & documentation gaps
    safety_flags: List[SafetyReviewFlag]
    missing_information: List[MissingInformation]

    # 5. Draft clinical documentation
    draft_soap_note: Optional[DraftSOAPNote]

    # 6. Workflow control, audit & failure diagnostics
    workflow_status: str  # "pending" | "in_progress" | "completed" | "failed"
    error: Optional[str]
    error_stage: Optional[str]


# =====================================================================
# Dependency Injection Container
# =====================================================================

class ClinicalWorkflowServices:
    """
    Dependency injection container for MediTrace clinical workflow services.
    Enables injecting mock LLM, embedding, vector store, and rule engines
    for 100% offline, deterministic testing without API keys or live network calls.
    """

    def __init__(
        self,
        extraction_service: Optional[ClinicalExtractionService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        safety_service: Optional[ClinicalSafetyVerificationService] = None,
        soap_service: Optional[SOAPNoteService] = None,
    ):
        self._extraction_service = extraction_service
        self._retrieval_service = retrieval_service
        self._safety_service = safety_service
        self._soap_service = soap_service

    @property
    def extraction_service(self) -> ClinicalExtractionService:
        if self._extraction_service is None:
            self._extraction_service = ClinicalExtractionService()
        return self._extraction_service

    @property
    def retrieval_service(self) -> Optional[RetrievalService]:
        if self._retrieval_service is None:
            try:
                self._retrieval_service = RetrievalService()
            except Exception as exc:
                logger.warning(
                    "Default RetrievalService initialization deferred: %s",
                    type(exc).__name__,
                )
                return None
        return self._retrieval_service

    @property
    def safety_service(self) -> ClinicalSafetyVerificationService:
        if self._safety_service is None:
            self._safety_service = ClinicalSafetyVerificationService()
        return self._safety_service

    @property
    def soap_service(self) -> SOAPNoteService:
        if self._soap_service is None:
            self._soap_service = SOAPNoteService(
                safety_service=self._safety_service
            )
        return self._soap_service


# =====================================================================
# Graph Construction Factory
# =====================================================================

def create_clinical_graph(services: Optional[ClinicalWorkflowServices] = None):
    """
    Build and compile the LangGraph StateGraph orchestrating Member A's clinical AI services.

    Args:
        services: Optional ClinicalWorkflowServices container with pre-configured or mocked services.

    Returns:
        Compiled LangGraph workflow runnable.
    """
    resolved_services = services or ClinicalWorkflowServices()

    # -----------------------------------------------------------------
    # Stage A: Validate Input
    # -----------------------------------------------------------------
    def validate_input(state: ClinicalState) -> ClinicalState:
        """Validate presence of dialogue transcript and encounter identifier."""
        transcript = state.get("transcript")
        if not transcript or not isinstance(transcript, str) or len(transcript.strip()) < 10:
            return {
                **state,
                "workflow_status": "failed",
                "error": "Consultation transcript must be a non-empty string of at least 10 characters.",
                "error_stage": "validate_input",
            }

        raw_id = state.get("encounter_id")
        encounter_id = (
            str(raw_id).strip()
            if raw_id and str(raw_id).strip()
            else f"enc_{uuid.uuid4().hex[:8]}"
        )

        return {
            **state,
            "encounter_id": encounter_id,
            "transcript": transcript.strip(),
            "known_allergies": list(state.get("known_allergies") or []),
            "current_medications": list(state.get("current_medications") or []),
            "existing_conditions": list(state.get("existing_conditions") or []),
            "workflow_status": "in_progress",
        }

    # -----------------------------------------------------------------
    # Stage B: Clinical Information Extraction
    # -----------------------------------------------------------------
    def extract_clinical_information(state: ClinicalState) -> ClinicalState:
        """Call ClinicalExtractionService to extract structured entities and data gaps."""
        if state.get("error"):
            return state

        try:
            extraction_svc = resolved_services.extraction_service
            result = extraction_svc.extract_clinical_information(
                transcript=state["transcript"],
                encounter_id=state.get("encounter_id"),
            )

            if not isinstance(result, ClinicalExtractionResult):
                return {
                    **state,
                    "workflow_status": "failed",
                    "error": "Clinical extraction returned unexpected schema type.",
                    "error_stage": "extract_clinical_information",
                }

            # Merge known intake allergies into extraction result if not present in transcript
            intake_allergies = state.get("known_allergies") or []
            if intake_allergies:
                existing_allergens = {
                    a.allergen.lower().strip() for a in result.allergies if a.allergen
                }
                for allergen in intake_allergies:
                    if allergen and allergen.strip() and allergen.lower().strip() not in existing_allergens:
                        result.allergies.append(
                            AllergyInfo(
                                allergen=allergen.strip(),
                                source=InformationSource.PATIENT_REPORTED,
                                verified_by_clinician=False,
                            )
                        )

            # Merge known intake medications into extraction result if not present
            intake_meds = state.get("current_medications") or []
            if intake_meds:
                existing_meds = {
                    m.name.lower().strip() for m in result.medications if m.name
                }
                for med in intake_meds:
                    if med and med.strip() and med.lower().strip() not in existing_meds:
                        result.medications.append(
                            MedicationMention(
                                name=med.strip(),
                                status=MedicationStatus.CURRENT,
                            )
                        )

            return {
                **state,
                "extraction_result": result,
                "missing_information": list(result.missing_information or []),
            }

        except Exception as exc:
            logger.error("Clinical extraction node failed: %s: %s", type(exc).__name__, exc)
            return {
                **state,
                "workflow_status": "failed",
                "error": f"Clinical extraction failed: {type(exc).__name__}: {str(exc)}",
                "error_stage": "extract_clinical_information",
            }

    # -----------------------------------------------------------------
    # Stage C: Clinical Query Formulation
    # -----------------------------------------------------------------
    def formulate_retrieval_query(state: ClinicalState) -> ClinicalState:
        """Synthesize a clinically targeted semantic query from extracted entities without PII."""
        if state.get("error"):
            return state

        extraction = state.get("extraction_result")
        if not extraction:
            return {
                **state,
                "workflow_status": "failed",
                "error": "Extraction result is required for retrieval query formulation.",
                "error_stage": "formulate_retrieval_query",
            }

        # Collect distinct clinical terms without patient identifiers
        med_names = [
            m.name.strip() for m in extraction.medications if m.name and m.name.strip()
        ]
        allergen_names = [
            a.allergen.strip() for a in extraction.allergies if a.allergen and a.allergen.strip()
        ]
        symptom_names = [
            s.name.strip()
            for s in extraction.symptoms
            if s.name and s.name.strip() and s.status == PresenceStatus.PRESENT
        ]

        # Prioritize drug-allergy interactions, medications, and chief complaints
        if med_names and allergen_names:
            query = f"{' '.join(med_names)} {' '.join(allergen_names)} allergy cross reactivity contraindication"
        elif med_names:
            query = f"{' '.join(med_names)} contraindications dosing clinical guidelines"
        elif symptom_names:
            query = f"{' '.join(symptom_names[:3])} evaluation management clinical guidelines"
        else:
            query = "clinical practice guidelines evaluation"

        return {
            **state,
            "retrieval_query": query.strip(),
        }

    # -----------------------------------------------------------------
    # Stage D: RAG Evidence Retrieval
    # -----------------------------------------------------------------
    def retrieve_references(state: ClinicalState) -> ClinicalState:
        """Query Qdrant vector store via RetrievalService for clinical guideline evidence."""
        if state.get("error"):
            return state

        query = state.get("retrieval_query")
        retrieval_svc = resolved_services.retrieval_service

        if not query or retrieval_svc is None:
            return {
                **state,
                "reference_passages": [],
                "retrieval_status": (
                    "unavailable_error" if retrieval_svc is None else "no_evidence_found"
                ),
            }

        try:
            passages = retrieval_svc.retrieve(query=query)
            if passages and len(passages) > 0:
                return {
                    **state,
                    "reference_passages": passages,
                    "retrieval_status": "success",
                }
            return {
                **state,
                "reference_passages": [],
                "retrieval_status": "no_evidence_found",
            }
        except Exception as exc:
            logger.warning("RetrievalService execution encountered an error: %s", type(exc).__name__)
            return {
                **state,
                "reference_passages": [],
                "retrieval_status": "unavailable_error",
            }

    # -----------------------------------------------------------------
    # Stage E: Clinical Safety Verification
    # -----------------------------------------------------------------
    def verify_safety(state: ClinicalState) -> ClinicalState:
        """Execute deterministic offline clinical safety verification rules."""
        if state.get("error"):
            return state

        extraction = state.get("extraction_result")
        if not extraction:
            return {
                **state,
                "workflow_status": "failed",
                "error": "Extraction result is required for safety verification.",
                "error_stage": "verify_safety",
            }

        try:
            safety_svc = resolved_services.safety_service
            references = state.get("reference_passages") or []
            flags = safety_svc.verify(
                extraction_result=extraction,
                reference_passages=references if references else None,
            )

            if not isinstance(flags, list) or any(not isinstance(f, SafetyReviewFlag) for f in flags):
                return {
                    **state,
                    "workflow_status": "failed",
                    "error": "Safety verification returned invalid flag types.",
                    "error_stage": "verify_safety",
                }

            return {
                **state,
                "safety_flags": flags,
            }

        except Exception as exc:
            logger.error("Safety verification node failed: %s: %s", type(exc).__name__, exc)
            return {
                **state,
                "workflow_status": "failed",
                "error": f"Safety verification failed: {type(exc).__name__}: {str(exc)}",
                "error_stage": "verify_safety",
            }

    # -----------------------------------------------------------------
    # Stage F: Draft SOAP Note Synthesis
    # -----------------------------------------------------------------
    def generate_soap_draft(state: ClinicalState) -> ClinicalState:
        """Call SOAPNoteService to synthesize structured draft SOAP documentation."""
        if state.get("error"):
            return state

        extraction = state.get("extraction_result")
        if not extraction:
            return {
                **state,
                "workflow_status": "failed",
                "error": "Extraction result is required for SOAP note generation.",
                "error_stage": "generate_soap_draft",
            }

        try:
            soap_svc = resolved_services.soap_service
            flags = state.get("safety_flags") or []
            references = state.get("reference_passages") or []
            transcript = state.get("transcript")
            encounter_id = state.get("encounter_id", "unspecified")

            draft_note = soap_svc.generate_soap_note(
                extraction_result=extraction,
                safety_flags=flags,
                reference_passages=references,
                transcript=transcript,
                note_id=f"soap_{encounter_id}",
            )

            if not isinstance(draft_note, DraftSOAPNote):
                return {
                    **state,
                    "workflow_status": "failed",
                    "error": "SOAP note service did not return a valid DraftSOAPNote instance.",
                    "error_stage": "generate_soap_draft",
                }

            return {
                **state,
                "draft_soap_note": draft_note,
            }

        except Exception as exc:
            logger.error("SOAP note generation node failed: %s: %s", type(exc).__name__, exc)
            return {
                **state,
                "workflow_status": "failed",
                "error": f"SOAP note generation failed: {type(exc).__name__}: {str(exc)}",
                "error_stage": "generate_soap_draft",
            }

    # -----------------------------------------------------------------
    # Stage G: Finalize Response
    # -----------------------------------------------------------------
    def finalize_response(state: ClinicalState) -> ClinicalState:
        """Compile final status and ensure error or completion state is consistent."""
        if state.get("error"):
            state["workflow_status"] = "failed"
        else:
            state["workflow_status"] = "completed"
        return state

    # -----------------------------------------------------------------
    # Routing Conditions
    # -----------------------------------------------------------------
    def route_after_validation(state: ClinicalState) -> str:
        if state.get("error"):
            return "finalize_response"
        return "extract_clinical_information"

    def route_after_extraction(state: ClinicalState) -> str:
        if state.get("error") or not state.get("extraction_result"):
            return "finalize_response"
        return "formulate_retrieval_query"

    def route_after_safety(state: ClinicalState) -> str:
        if state.get("error"):
            return "finalize_response"
        return "generate_soap_draft"

    # -----------------------------------------------------------------
    # Build LangGraph StateGraph
    # -----------------------------------------------------------------
    graph = StateGraph(ClinicalState)

    # Register Nodes
    graph.add_node("validate_input", validate_input)
    graph.add_node("extract_clinical_information", extract_clinical_information)
    graph.add_node("formulate_retrieval_query", formulate_retrieval_query)
    graph.add_node("retrieve_references", retrieve_references)
    graph.add_node("verify_safety", verify_safety)
    graph.add_node("generate_soap_draft", generate_soap_draft)
    graph.add_node("finalize_response", finalize_response)

    # Set Entry Point
    graph.set_entry_point("validate_input")

    # Add Conditional & Static Edges
    graph.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "extract_clinical_information": "extract_clinical_information",
            "finalize_response": "finalize_response",
        },
    )

    graph.add_conditional_edges(
        "extract_clinical_information",
        route_after_extraction,
        {
            "formulate_retrieval_query": "formulate_retrieval_query",
            "finalize_response": "finalize_response",
        },
    )

    graph.add_edge("formulate_retrieval_query", "retrieve_references")
    graph.add_edge("retrieve_references", "verify_safety")

    graph.add_conditional_edges(
        "verify_safety",
        route_after_safety,
        {
            "generate_soap_draft": "generate_soap_draft",
            "finalize_response": "finalize_response",
        },
    )

    graph.add_edge("generate_soap_draft", "finalize_response")
    graph.add_edge("finalize_response", END)

    return graph.compile()


# =====================================================================
# Workflow Invocation Interface
# =====================================================================

class ClinicalWorkflow:
    """
    High-level clinical copilot workflow runner encapsulating the LangGraph graph.
    Provides dependency injection for testing and structured execution.
    """

    def __init__(self, services: Optional[ClinicalWorkflowServices] = None):
        self.services = services or ClinicalWorkflowServices()
        self.compiled_graph = create_clinical_graph(self.services)

    def invoke(self, input_data: Union[Dict[str, Any], ClinicalState]) -> ClinicalState:
        """
        Execute the end-to-end clinical workflow on consultation input.

        Args:
            input_data: Dict or ClinicalState containing at minimum 'transcript'.

        Returns:
            ClinicalState with extraction result, retrieved evidence, safety flags, and draft SOAP note.
        """
        encounter_id = input_data.get("encounter_id") or f"enc_{uuid.uuid4().hex[:8]}"

        initial_state: ClinicalState = {
            "encounter_id": str(encounter_id).strip(),
            "transcript": input_data.get("transcript", ""),
            "patient_id": input_data.get("patient_id"),
            "patient_name": input_data.get("patient_name"),
            "patient_age": input_data.get("patient_age"),
            "patient_gender": input_data.get("patient_gender"),
            "known_allergies": list(input_data.get("known_allergies") or []),
            "current_medications": list(input_data.get("current_medications") or []),
            "existing_conditions": list(input_data.get("existing_conditions") or []),
            "extraction_result": None,
            "retrieval_query": None,
            "reference_passages": [],
            "retrieval_status": "not_attempted",
            "safety_flags": [],
            "missing_information": [],
            "draft_soap_note": None,
            "workflow_status": "pending",
            "error": None,
            "error_stage": None,
        }

        result = self.compiled_graph.invoke(initial_state)
        return result
