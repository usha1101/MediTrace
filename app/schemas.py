"""
MediTrace - Clinical AI Engine Schemas
Ambient Clinical Copilot with Real-Time Safety Verification

CLINICAL SAFETY NOTICE:
-----------------------
MediTrace is an educational clinical documentation prototype.
It must NOT autonomously diagnose, prescribe, or make treatment decisions.
All draft clinical documentation, extracted findings, and safety flags are generated
for qualified clinician review and validation only.
The absence of an alert or flag must NEVER be interpreted as proof that a patient is safe.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =====================================================================
# Constants & Clinical Disclaimers
# =====================================================================

CLINICAL_SAFETY_DISCLAIMER: str = (
    "EDUCATIONAL PROTOTYPE ONLY: MediTrace does not autonomously diagnose, "
    "prescribe, or make clinical treatment decisions. All extracted findings, "
    "draft SOAP notes, and safety-review flags require review and verification "
    "by a qualified, licensed clinician. The absence of an alert must never "
    "be interpreted as proof that a patient is safe."
)


# =====================================================================
# Provenance, Uncertainty, and Status Enums
# =====================================================================

class InformationSource(str, Enum):
    """
    Distinguishes the clinical provenance of data.
    Separates subjective patient statements from objective clinical measurements
    or verified electronic health records (EHR).
    """
    PATIENT_REPORTED = "patient_reported"
    CLINICIAN_ENTERED = "clinician_entered"
    INDEPENDENTLY_VERIFIED = "independently_verified"
    HISTORICAL_EHR = "historical_ehr"
    UNKNOWN = "unknown"


class CertaintyLevel(str, Enum):
    """
    Preserves uncertainty in clinical statements without premature closure.
    """
    CONFIRMED = "confirmed"
    SUSPECTED = "suspected"
    UNCERTAIN = "uncertain"
    DENIED = "denied"


class PresenceStatus(str, Enum):
    """
    Explicitly indicates whether a clinical entity is present, negated, or unknown.
    """
    PRESENT = "present"
    ABSENT = "absent"  # Explicitly negated by patient or clinician
    UNKNOWN = "unknown"  # Ambiguous or unstated


class MedicationStatus(str, Enum):
    """
    Prescribes the current status of a mentioned medication.
    """
    CURRENT = "current"
    DISCONTINUED = "discontinued"
    PROPOSED = "proposed"
    DENIED = "denied"
    UNKNOWN = "unknown"


class SafetySeverity(str, Enum):
    """
    Triage severity level for safety-review flags.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyCategory(str, Enum):
    """
    Categorization of clinical safety verification checks.
    """
    DRUG_ALLERGY_CONFLICT = "drug_allergy_conflict"
    DRUG_DRUG_INTERACTION = "drug_drug_interaction"
    CONTRAINDICATION = "contraindication"
    DOSAGE_WARNING = "dosage_warning"
    MISSING_VITAL_INFO = "missing_vital_info"
    RED_FLAG_SYMPTOM = "red_flag_symptom"
    DIAGNOSTIC_DISCREPANCY = "diagnostic_discrepancy"
    OTHER = "other"


class ReviewStatus(str, Enum):
    """
    Human-in-the-loop clinician acknowledgment and verification status.
    """
    PENDING_REVIEW = "pending_review"
    ACCEPTED = "accepted"
    DISMISSED_AS_INTENTIONAL = "dismissed_as_intentional"
    NOT_APPLICABLE = "not_applicable"


class NoteStatus(str, Enum):
    """
    Workflow lifecycle state of a generated clinical document.
    """
    DRAFT_PENDING_REVIEW = "draft_pending_review"
    CLINICIAN_EDITED = "clinician_edited"
    CLINICIAN_APPROVED = "clinician_approved"
    REJECTED = "rejected"


# =====================================================================
# 1. Symptom Information
# =====================================================================

class SymptomInfo(BaseModel):
    """
    Model representing a symptom mentioned in clinical consultation.
    Preserves presence status, clinical certainty, and provenance.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Name or description of the symptom")
    status: PresenceStatus = Field(
        default=PresenceStatus.PRESENT,
        description="Whether the symptom is present, absent (denied), or unknown"
    )
    certainty: CertaintyLevel = Field(
        default=CertaintyLevel.UNCERTAIN,
        description="Clinician or patient degree of certainty (defaults to uncertain)"
    )
    source: InformationSource = Field(
        default=InformationSource.PATIENT_REPORTED,
        description="Provenance of the symptom mention"
    )
    body_site: Optional[str] = Field(default=None, description="Anatomical location if specified")
    duration: Optional[str] = Field(default=None, description="Duration or timecourse if reported")
    severity: Optional[str] = Field(default=None, description="Self-reported or observed severity")
    raw_quote: Optional[str] = Field(
        default=None,
        description="Exact quote or phrase from transcript to prevent hallucinated details"
    )
    notes: Optional[str] = Field(default=None, description="Additional context or notes")

    @field_validator("name")
    @classmethod
    def validate_name_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Symptom name must not be empty or whitespace only.")
        return v.strip()

    @model_validator(mode="after")
    def validate_presence_and_certainty(self) -> "SymptomInfo":
        # Clinical logic: cannot assert confirmed certainty when status is unknown
        if self.status == PresenceStatus.UNKNOWN and self.certainty == CertaintyLevel.CONFIRMED:
            raise ValueError(
                "Inconsistent symptom certainty: Cannot set certainty to CONFIRMED when presence status is UNKNOWN."
            )
        return self


# =====================================================================
# 2. Patient-Reported Allergies
# =====================================================================

class AllergyInfo(BaseModel):
    """
    Model representing an allergy or adverse reaction report.
    Carefully distinguishes patient-reported allergies from independently verified allergy records.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    allergen: str = Field(..., description="Causative agent, drug, food, or substance")
    reaction: Optional[str] = Field(
        default=None,
        description="Observed or reported allergic manifestation (e.g., rash, hives, anaphylaxis)"
    )
    reaction_severity: Optional[str] = Field(
        default=None,
        description="Severity of reaction if stated (e.g., mild, severe, life-threatening)"
    )
    source: InformationSource = Field(
        default=InformationSource.PATIENT_REPORTED,
        description="Origin of allergy report"
    )
    verified_by_clinician: bool = Field(
        default=False,
        description="True if independently verified by a clinician or medical record, False if unverified report"
    )
    raw_quote: Optional[str] = Field(
        default=None,
        description="Direct quote from the dialogue regarding this allergy"
    )
    notes: Optional[str] = Field(default=None, description="Additional clinical notes regarding allergy")

    @field_validator("allergen")
    @classmethod
    def validate_allergen_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Allergen must not be empty or whitespace only.")
        return v.strip()

    @model_validator(mode="after")
    def validate_source_and_verification_consistency(self) -> "AllergyInfo":
        if self.source == InformationSource.PATIENT_REPORTED and self.verified_by_clinician:
            raise ValueError(
                "Contradictory allergy verification: source cannot be 'patient_reported' when "
                "verified_by_clinician is True. Patient-reported reports are unverified statements. "
                "Update source to 'clinician_entered' or 'independently_verified' if clinically verified."
            )
        if self.source == InformationSource.INDEPENDENTLY_VERIFIED and not self.verified_by_clinician:
            raise ValueError(
                "Contradictory allergy verification: source is 'independently_verified' but "
                "verified_by_clinician is False. Independently verified records cannot be unverified."
            )
        if self.source == InformationSource.UNKNOWN and self.verified_by_clinician:
            raise ValueError(
                "Contradictory allergy verification: source cannot be 'unknown' when "
                "verified_by_clinician is True."
            )
        return self


# =====================================================================
# 3. Medications Mentioned in Consultation Text
# =====================================================================

class MedicationMention(BaseModel):
    """
    Model representing a medication mentioned in the consultation text.
    Attributes like dosage, route, and frequency default to None if not explicitly stated.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Medication brand or generic name")
    dosage: Optional[str] = Field(
        default=None,
        description="Dose (e.g., '500 mg'). Must remain None if unmentioned; never invent a dose."
    )
    route: Optional[str] = Field(
        default=None,
        description="Administration route (e.g., 'oral', 'IV'). Must remain None if unstated."
    )
    frequency: Optional[str] = Field(
        default=None,
        description="Frequency (e.g., 'once daily', 'PRN'). Must remain None if unstated."
    )
    duration: Optional[str] = Field(default=None, description="Course duration if mentioned")
    status: MedicationStatus = Field(
        default=MedicationStatus.UNKNOWN,
        description="Current usage state of the medication (defaults to unknown until established)"
    )
    source: InformationSource = Field(
        default=InformationSource.PATIENT_REPORTED,
        description="Origin of the medication information"
    )
    is_adherent: Optional[bool] = Field(
        default=None,
        description="Self-reported compliance (True=taking as prescribed, False=non-adherent, None=unasked)"
    )
    raw_quote: Optional[str] = Field(
        default=None,
        description="Direct verbatim quote from consultation"
    )

    @field_validator("name")
    @classmethod
    def validate_name_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Medication name must not be empty or whitespace only.")
        return v.strip()


# =====================================================================
# 4. Available Observations and Vital Signs
# =====================================================================

class VitalSigns(BaseModel):
    """
    Validated physiological vital signs with strict physiological boundary checks.
    Values outside physiologically viable ranges trigger explicit validation errors.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    blood_pressure_systolic: Optional[int] = Field(
        default=None,
        description="Systolic blood pressure in mmHg (valid range: 40-300)"
    )
    blood_pressure_diastolic: Optional[int] = Field(
        default=None,
        description="Diastolic blood pressure in mmHg (valid range: 20-200)"
    )
    heart_rate: Optional[int] = Field(
        default=None,
        description="Heart rate in beats per minute (valid range: 20-300)"
    )
    respiratory_rate: Optional[int] = Field(
        default=None,
        description="Respiratory rate in breaths per minute (valid range: 4-80)"
    )
    temperature_celsius: Optional[float] = Field(
        default=None,
        description="Core body temperature in Celsius (valid range: 28.0-45.0)"
    )
    oxygen_saturation_pct: Optional[float] = Field(
        default=None,
        description="Pulse oximetry SpO2 percentage (valid range: 40.0-100.0)"
    )
    source: InformationSource = Field(
        default=InformationSource.CLINICIAN_ENTERED,
        description="Provenance of vital sign measurements"
    )
    measurement_time: Optional[str] = Field(default=None, description="Timestamp or recorded time")
    notes: Optional[str] = Field(default=None, description="Clinician notes on vitals measurement")

    @field_validator("blood_pressure_systolic")
    @classmethod
    def validate_systolic_bp(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (40 <= v <= 300):
            raise ValueError(
                f"Systolic blood pressure ({v} mmHg) is outside physiologically plausible range (40-300 mmHg). "
                "Unusual or extreme values must be reviewed by a clinician rather than recorded as valid vital signs."
            )
        return v

    @field_validator("blood_pressure_diastolic")
    @classmethod
    def validate_diastolic_bp(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (20 <= v <= 200):
            raise ValueError(
                f"Diastolic blood pressure ({v} mmHg) is outside physiologically plausible range (20-200 mmHg). "
                "Unusual or extreme values must be reviewed by a clinician rather than recorded as valid vital signs."
            )
        return v

    @field_validator("heart_rate")
    @classmethod
    def validate_heart_rate(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (20 <= v <= 300):
            raise ValueError(
                f"Heart rate ({v} bpm) is outside physiologically plausible range (20-300 bpm). "
                "Unusual or extreme values must be reviewed by a clinician rather than recorded as valid vital signs."
            )
        return v

    @field_validator("respiratory_rate")
    @classmethod
    def validate_respiratory_rate(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (4 <= v <= 80):
            raise ValueError(
                f"Respiratory rate ({v} breaths/min) is outside physiologically plausible range (4-80 breaths/min). "
                "Unusual or extreme values must be reviewed by a clinician rather than recorded as valid vital signs."
            )
        return v

    @field_validator("temperature_celsius")
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (28.0 <= v <= 45.0):
            raise ValueError(
                f"Body temperature ({v} °C) is outside physiologically plausible range (28.0-45.0 °C). "
                "Unusual or extreme values must be reviewed by a clinician rather than recorded as valid vital signs."
            )
        return v

    @field_validator("oxygen_saturation_pct")
    @classmethod
    def validate_oxygen_saturation(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (40.0 <= v <= 100.0):
            raise ValueError(
                f"Oxygen saturation ({v}%) is outside physiologically plausible range (40.0-100.0%). "
                "Unusual or extreme values must be reviewed by a clinician rather than recorded as valid vital signs."
            )
        return v

    @model_validator(mode="after")
    def validate_blood_pressure_relationship(self) -> "VitalSigns":
        sys_bp = self.blood_pressure_systolic
        dia_bp = self.blood_pressure_diastolic
        if sys_bp is not None and dia_bp is not None:
            if sys_bp <= dia_bp:
                raise ValueError(
                    f"Invalid blood pressure: Systolic pressure ({sys_bp} mmHg) must be strictly "
                    f"greater than diastolic pressure ({dia_bp} mmHg)."
                )
        return self


class ObservationInfo(BaseModel):
    """
    Model representing physical examination findings, lab observations, or clinical notes.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    category: str = Field(
        ...,
        description="Observation category (e.g., 'physical_exam', 'laboratory', 'imaging')"
    )
    description: str = Field(..., description="Narrative clinical finding or observation")
    source: InformationSource = Field(
        default=InformationSource.CLINICIAN_ENTERED,
        description="Provenance of the observation"
    )
    is_normal: Optional[bool] = Field(
        default=None,
        description="True if within normal limits, False if abnormal, None if unassessed or undetermined"
    )
    raw_quote: Optional[str] = Field(default=None, description="Supporting dialogue or note snippet")

    @field_validator("category", "description")
    @classmethod
    def validate_non_blank_strings(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must not be empty or whitespace only.")
        return v.strip()


# =====================================================================
# 5. Missing Information
# =====================================================================

class MissingInformation(BaseModel):
    """
    Explicit representation of missing or unmentioned clinical data.
    Preventing omission errors and hallucination by recording what was NOT asked or documented.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    category: str = Field(
        ...,
        description="Domain of missing information (e.g., 'allergy_history', 'medication_dosage', 'vital_signs')"
    )
    description: str = Field(
        ...,
        description="Specific clinical detail that was omitted or not discussed in consultation"
    )
    clinical_relevance: str = Field(
        ...,
        description="Clinical rationale explaining why this omission is relevant to patient safety"
    )
    recommended_follow_up: Optional[str] = Field(
        default=None,
        description="Suggested inquiry or action for clinician review"
    )

    @field_validator("category", "description", "clinical_relevance")
    @classmethod
    def validate_required_fields(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must not be empty or whitespace only.")
        return v.strip()


# =====================================================================
# 7. Retrieved Reference Passages with Source Information
# =====================================================================

class ReferencePassage(BaseModel):
    """
    Model representing retrieved evidence from clinical guidelines, drug labels, or literature.
    Provides verifiable citations and provenance for RAG-assisted safety checks and notes.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    passage_id: str = Field(..., description="Unique identifier for the passage or chunk")
    title: str = Field(..., description="Title of the guideline or reference section")
    content: str = Field(..., description="Verbatim text content of the reference excerpt")
    source_document: str = Field(..., description="Source publication (e.g., 'NICE NG28', 'FDA Drug Label')")
    source_url_or_doi: Optional[str] = Field(default=None, description="Authoritative URL, DOI, or URI")
    section_or_page: Optional[str] = Field(default=None, description="Section heading or page number")
    relevance_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Retrieval similarity score bounded between 0.0 and 1.0"
    )
    retrieval_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional vector search metadata"
    )

    @field_validator("passage_id", "title", "content", "source_document")
    @classmethod
    def validate_non_blank_strings(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must not be empty or whitespace only.")
        return v.strip()


class DocumentChunk(BaseModel):
    """
    Model representing a deterministic reference text chunk for RAG ingestion.
    Preserves document provenance, section context, and metadata without fabricating missing details.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    passage_id: str = Field(..., description="Unique deterministic identifier for the passage or chunk")
    document_id: str = Field(..., description="Unique identifier of the parent reference document")
    text: str = Field(..., description="Verbatim normalized text content of this chunk")
    source_title: str = Field(..., description="Title of the source publication, guideline, or label")
    source_type: str = Field(
        ...,
        description="Classification of the source (e.g., 'clinical_guideline', 'drug_label', 'protocol')"
    )
    section: Optional[str] = Field(default=None, description="Section heading or structural location if available")
    publication_date: Optional[str] = Field(
        default=None,
        description="Publication or revision date if explicitly available in source metadata"
    )
    chunk_index: int = Field(default=0, ge=0, description="0-based sequential chunk index within the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provenance or chunking metadata"
    )

    @field_validator("passage_id", "document_id", "text", "source_title", "source_type")
    @classmethod
    def validate_non_blank_fields(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must not be empty or whitespace only.")
        return v.strip()

    def to_reference_passage(self, relevance_score: Optional[float] = None) -> ReferencePassage:
        """
        Convert this DocumentChunk into a ReferencePassage for SOAP notes and safety review flags.
        """
        title_with_section = f"{self.source_title} - {self.section}" if self.section else self.source_title
        return ReferencePassage(
            passage_id=self.passage_id,
            title=title_with_section,
            content=self.text,
            source_document=self.source_title,
            section_or_page=self.section,
            relevance_score=relevance_score,
            retrieval_metadata={
                **self.metadata,
                "document_id": self.document_id,
                "source_type": self.source_type,
                "chunk_index": self.chunk_index,
                "publication_date": self.publication_date,
            },
        )


# Aliases for flexible naming across RAG components
ReferencePassageChunk = DocumentChunk
DocumentPassage = DocumentChunk


# =====================================================================
# 8. Potential Safety-Review Flags
# =====================================================================

class SafetyReviewFlag(BaseModel):
    """
    Model representing a clinical safety alert flagged for clinician review.
    All flags require human-in-the-loop acknowledgment before clinical signoff.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    flag_id: str = Field(..., description="Unique identifier for the safety alert")
    category: SafetyCategory = Field(..., description="Type of clinical safety check triggered")
    severity: SafetySeverity = Field(..., description="Triage severity level")
    title: str = Field(..., description="Summary headline of the safety flag")
    detail: str = Field(..., description="Detailed clinical description of the potential risk")
    flagged_items: List[str] = Field(
        default_factory=list,
        description="Specific entities triggering the flag (e.g., ['Penicillin', 'Amoxicillin'])"
    )
    supporting_references: List[ReferencePassage] = Field(
        default_factory=list,
        description="Retrieved guideline passages supporting this flag"
    )
    requires_clinician_acknowledgment: bool = Field(
        default=True,
        description="Must be acknowledged by the clinician before finalizing documentation"
    )
    clinician_review_status: ReviewStatus = Field(
        default=ReviewStatus.PENDING_REVIEW,
        description="Clinician review and disposition status"
    )
    clinician_notes: Optional[str] = Field(
        default=None,
        description="Free text comments or justification from the reviewing clinician"
    )
    disclaimer: str = Field(
        default=CLINICAL_SAFETY_DISCLAIMER,
        description="Mandatory prototype safety notice"
    )

    @field_validator("flag_id", "title", "detail")
    @classmethod
    def validate_non_blank_strings(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} must not be empty or whitespace only.")
        return v.strip()

    @field_validator("disclaimer")
    @classmethod
    def validate_disclaimer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Clinical safety disclaimer must not be cleared or empty.")
        return v.strip()


# =====================================================================
# 6. Clinical Extraction Results
# =====================================================================

class ClinicalExtractionResult(BaseModel):
    """
    Aggregated structured extraction output from raw ambient consultation text.
    Combines symptoms, allergies, medications, observations, vitals, and missing information.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    encounter_id: Optional[str] = Field(default=None, description="Optional consultation or encounter identifier")
    symptoms: List[SymptomInfo] = Field(default_factory=list, description="Extracted symptom entities")
    allergies: List[AllergyInfo] = Field(default_factory=list, description="Extracted allergy reports")
    medications: List[MedicationMention] = Field(default_factory=list, description="Extracted medication mentions")
    vitals: Optional[VitalSigns] = Field(default=None, description="Extracted vital signs if measured")
    observations: List[ObservationInfo] = Field(
        default_factory=list,
        description="Extracted physical exam findings and observations"
    )
    missing_information: List[MissingInformation] = Field(
        default_factory=list,
        description="Explicit gaps in documentation or unasked questions"
    )
    consultation_summary: Optional[str] = Field(
        default=None,
        description="Draft summary synthesizing discussion without adding unstated facts"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extraction metadata such as timestamps and model versions"
    )
    disclaimer: str = Field(
        default=CLINICAL_SAFETY_DISCLAIMER,
        description="Mandatory clinical prototype safety notice"
    )

    @field_validator("disclaimer")
    @classmethod
    def validate_disclaimer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Clinical safety disclaimer must not be cleared or empty.")
        return v.strip()


# =====================================================================
# 9. Draft SOAP Notes
# =====================================================================

class DraftSOAPNote(BaseModel):
    """
    Standardized draft clinical progress note structured into:
    Subjective, Objective, Assessment, and Plan (SOAP).
    Must remain in DRAFT status until explicitly verified and signed off by a clinician.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    note_id: Optional[str] = Field(default=None, description="Unique identifier for the SOAP note")
    subjective: str = Field(
        ...,
        description="Subjective: Chief complaint, history of present illness, patient-reported symptoms"
    )
    objective: str = Field(
        ...,
        description="Objective: Vital signs, physical examination, and reviewed objective data"
    )
    assessment: str = Field(
        ...,
        description="Assessment: Clinical synthesis, differential considerations, and status evaluation"
    )
    plan: str = Field(
        ...,
        description="Plan: Proposed diagnostics, management, therapeutics, patient education, and follow-up"
    )
    safety_flags: List[SafetyReviewFlag] = Field(
        default_factory=list,
        description="Safety review flags linked to this encounter for clinician attention"
    )
    missing_information: List[MissingInformation] = Field(
        default_factory=list,
        description="Clinical data gaps to review before finalizing care"
    )
    references: List[ReferencePassage] = Field(
        default_factory=list,
        description="Retrieved guideline passages supporting the draft note"
    )
    status: NoteStatus = Field(
        default=NoteStatus.DRAFT_PENDING_REVIEW,
        description="Lifecycle status: starts as draft pending clinician review"
    )
    clinician_feedback: Optional[str] = Field(
        default=None,
        description="Clinician edits, corrections, or signoff commentary"
    )
    disclaimer: str = Field(
        default=CLINICAL_SAFETY_DISCLAIMER,
        description="Mandatory prototype safety disclaimer"
    )

    @field_validator("subjective", "objective", "assessment", "plan")
    @classmethod
    def validate_soap_sections_not_blank(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"SOAP section '{info.field_name}' must not be empty or whitespace only.")
        return v.strip()

    @field_validator("disclaimer")
    @classmethod
    def validate_disclaimer_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Clinical safety disclaimer must not be cleared or empty.")
        return v.strip()
