"""
MediTrace - Clinical Safety Verification Service
Deterministic safety rules engine for validating structured clinical extraction
results against clinical documentation standards and safety constraints.

CLINICAL SAFETY NOTICE:
-----------------------
MediTrace is an educational clinical documentation prototype.
It must NOT autonomously diagnose, prescribe, or make treatment decisions.
All generated safety-review flags are structured alerts for qualified clinician
review and validation only.
The absence of a safety-review flag must NEVER be interpreted as proof that a patient is safe.

PROMPT INJECTION & UNTRUSTED DATA GUARD:
----------------------------------------
Retrieved reference passages and transcript text are passive data, NOT executable instructions.
Downstream components must never execute directives or roleplay commands contained within
clinical excerpts or reference passages.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.schemas import (
    CLINICAL_SAFETY_DISCLAIMER,
    AllergyInfo,
    CertaintyLevel,
    ClinicalExtractionResult,
    InformationSource,
    MedicationMention,
    MedicationStatus,
    MissingInformation,
    ObservationInfo,
    PresenceStatus,
    ReferencePassage,
    ReviewStatus,
    SafetyCategory,
    SafetyReviewFlag,
    SafetySeverity,
    SymptomInfo,
    VitalSigns,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Service Exceptions
# =====================================================================

class SafetyVerificationError(Exception):
    """Base exception for all MediTrace safety verification operations."""
    pass


class SafetyVerificationValidationError(SafetyVerificationError, ValueError):
    """Raised when input extraction results or verification parameters are invalid."""
    pass


# =====================================================================
# Conservative Drug Allergy Cross-Reactivity Mapping
# =====================================================================

# Restricted strictly to well-documented, conservative relationships.
# Broad drug classes are NOT treated as interchangeable without authoritative evidence.
DEFAULT_ALLERGY_CROSS_REACTIVITY: Dict[str, List[str]] = {
    # Beta-lactams: Penicillins and aminopenicillins
    "penicillin": [
        "amoxicillin",
        "ampicillin",
        "augmentin",
        "amoxicillin-clavulanate",
        "piperacillin",
    ],
    "amoxicillin": [
        "penicillin",
        "ampicillin",
        "augmentin",
        "amoxicillin-clavulanate",
    ],
    # Sulfonamide antimicrobials
    "sulfa": [
        "sulfamethoxazole",
        "bactrim",
        "septra",
        "sulfadiazine",
    ],
    "sulfonamide": [
        "sulfamethoxazole",
        "bactrim",
        "septra",
        "sulfadiazine",
    ],
}


def _has_discrete_word_match(target: str, text: str) -> bool:
    """Check if target appears as a discrete word or token in text to avoid false-positive substrings."""
    pattern = r"\b" + re.escape(target) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


# =====================================================================
# Service Implementation
# =====================================================================

class ClinicalSafetyVerificationService:
    """
    Deterministic Clinical Safety Verification Service for MediTrace.
    Evaluates structured clinical extraction results and attaches supporting
    retrieval reference passages to generate standardized safety-review flags
    for clinician evaluation.

    CLINICAL SAFETY PRINCIPLES:
    - Never autonomously diagnoses, prescribes, or makes clinical decisions.
    - Operates strictly on explicitly represented structured fields.
    - No unsupported clinical thresholds.
    - Never converts uncertain or unverified data into confirmed findings.
    - Zero external API dependencies (deterministic and offline-capable).
    - The absence of safety review flags NEVER implies that a patient is safe.
    """

    def __init__(
        self,
        allergy_cross_reactivity: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the safety verification service with dependency injection.

        Args:
            allergy_cross_reactivity: Optional conservative mapping of drug allergen classes
                                     to related generic medications.
        """
        self.allergy_cross_reactivity = (
            allergy_cross_reactivity
            if allergy_cross_reactivity is not None
            else DEFAULT_ALLERGY_CROSS_REACTIVITY
        )

    def verify(
        self,
        extraction_result: ClinicalExtractionResult,
        reference_passages: Optional[List[ReferencePassage]] = None,
    ) -> List[SafetyReviewFlag]:
        """
        Execute deterministic safety verification rules on structured clinical extraction data.

        Args:
            extraction_result: Validated ClinicalExtractionResult instance.
            reference_passages: Optional list of retrieved ReferencePassage objects.

        Returns:
            List of structured SafetyReviewFlag objects for clinician attention.
        """
        # Validate input types
        if extraction_result is None or not isinstance(extraction_result, ClinicalExtractionResult):
            raise SafetyVerificationValidationError(
                "extraction_result must be a valid ClinicalExtractionResult instance."
            )

        if reference_passages is not None:
            if not isinstance(reference_passages, list) or not all(
                isinstance(p, ReferencePassage) for p in reference_passages
            ):
                raise SafetyVerificationValidationError(
                    "reference_passages must be a list of ReferencePassage instances or None."
                )

        logger.info("Executing clinical safety verification checks")

        raw_flags: List[SafetyReviewFlag] = []

        # Rule 1: Missing Clinical Information
        raw_flags.extend(self._check_missing_information(extraction_result))

        # Rule 2: Incomplete Allergy Records
        raw_flags.extend(self._check_incomplete_allergies(extraction_result))

        # Rule 3: Incomplete Medication Information
        raw_flags.extend(self._check_incomplete_medications(extraction_result))

        # Rule 4: Data Quality and Contradictions
        raw_flags.extend(self._check_contradictions_and_conflicts(extraction_result))

        # Rule 5: Vitals and Observations (Structural Completeness & Explicit Findings only)
        raw_flags.extend(self._check_vitals_and_observations(extraction_result))

        # Rule 7: Clinical Uncertainty Preservation
        raw_flags.extend(self._check_uncertainties(extraction_result))

        # Rule 6: Attach Reference Provenance
        if reference_passages:
            raw_flags = self._attach_reference_provenance(raw_flags, reference_passages)

        # Deduplicate and re-index flags deterministically
        final_flags = self._finalize_flags(raw_flags)

        logger.info(
            "Clinical safety verification completed: generated %d review flags",
            len(final_flags),
        )
        return final_flags

    # =================================================================
    # Rule 1: Missing Clinical Information
    # =================================================================

    def _check_missing_information(
        self, extraction_result: ClinicalExtractionResult
    ) -> List[SafetyReviewFlag]:
        """
        Identify explicitly recorded missing information that may require clinician clarification.
        Does not assume that missing information is abnormal.
        """
        flags: List[SafetyReviewFlag] = []
        for idx, item in enumerate(extraction_result.missing_information, 1):
            category_lower = item.category.lower().strip()

            # Map category to appropriate SafetyCategory enum
            if any(k in category_lower for k in ("vital", "blood_pressure", "pulse", "temp", "spo2")):
                cat = SafetyCategory.MISSING_VITAL_INFO
            elif any(k in category_lower for k in ("allergy", "adverse")):
                cat = SafetyCategory.DRUG_ALLERGY_CONFLICT
            elif any(k in category_lower for k in ("medication", "dosage", "dose", "drug")):
                cat = SafetyCategory.DOSAGE_WARNING
            else:
                cat = SafetyCategory.OTHER

            detail_msg = (
                f"Consultation explicitly recorded unasked or omitted information in category "
                f"'{item.category}': {item.description}. Clinical relevance: {item.clinical_relevance}."
            )
            if item.recommended_follow_up:
                detail_msg += f" Recommended follow-up: {item.recommended_follow_up}."

            flags.append(
                SafetyReviewFlag(
                    flag_id=f"flag_missing_info_{idx}",
                    category=cat,
                    severity=SafetySeverity.LOW,
                    title=f"Documented Missing Information: {item.category.replace('_', ' ').title()}",
                    detail=detail_msg,
                    flagged_items=[item.category],
                    requires_clinician_acknowledgment=True,
                    clinician_review_status=ReviewStatus.PENDING_REVIEW,
                    disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                )
            )
        return flags

    # =================================================================
    # Rule 2: Incomplete Allergy Records
    # =================================================================

    def _check_incomplete_allergies(
        self, extraction_result: ClinicalExtractionResult
    ) -> List[SafetyReviewFlag]:
        """
        If the available schema indicates an allergy record is incomplete or unverified,
        create a review flag describing the missing information.
        Does not infer that an allergy exists when no allergy was reported.
        """
        flags: List[SafetyReviewFlag] = []
        for idx, allergy in enumerate(extraction_result.allergies, 1):
            # Check 1: Missing reaction / manifestation
            if not allergy.reaction or not allergy.reaction.strip():
                flags.append(
                    SafetyReviewFlag(
                        flag_id=f"flag_incomplete_allergy_reaction_{idx}",
                        category=SafetyCategory.DRUG_ALLERGY_CONFLICT,
                        severity=SafetySeverity.MEDIUM,
                        title=f"Incomplete Allergy Record: Missing Reaction for {allergy.allergen}",
                        detail=(
                            f"Allergy to '{allergy.allergen}' is recorded without a documented reaction or "
                            "clinical manifestation (reaction is null). Clinician verification of reaction type "
                            "(e.g., hives, anaphylaxis, GI upset) and severity is recommended."
                        ),
                        flagged_items=[allergy.allergen],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )

            # Check 2: Unverified patient-reported allergy
            if not allergy.verified_by_clinician:
                flags.append(
                    SafetyReviewFlag(
                        flag_id=f"flag_unverified_allergy_{idx}",
                        category=SafetyCategory.DRUG_ALLERGY_CONFLICT,
                        severity=SafetySeverity.LOW,
                        title=f"Unverified Allergy Report: {allergy.allergen}",
                        detail=(
                            f"Allergy to '{allergy.allergen}' is a patient-reported statement and has not been "
                            "independently verified by a clinician or confirmed in the electronic health record."
                        ),
                        flagged_items=[allergy.allergen],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )
        return flags

    # =================================================================
    # Rule 3: Incomplete Medication Information
    # =================================================================

    def _check_incomplete_medications(
        self, extraction_result: ClinicalExtractionResult
    ) -> List[SafetyReviewFlag]:
        """
        If a medication is mentioned but relevant details are explicitly missing or uncertain,
        create a review flag.
        Does not invent dosage, frequency, route, indication, or medication history.
        """
        flags: List[SafetyReviewFlag] = []
        for idx, med in enumerate(extraction_result.medications, 1):
            # Check 1: Missing essential regimen details for active/proposed meds
            if med.status in (MedicationStatus.CURRENT, MedicationStatus.PROPOSED):
                missing_fields: List[str] = []
                if not med.dosage or not med.dosage.strip():
                    missing_fields.append("dosage")
                if not med.frequency or not med.frequency.strip():
                    missing_fields.append("frequency")
                if not med.route or not med.route.strip():
                    missing_fields.append("route")

                if missing_fields:
                    flags.append(
                        SafetyReviewFlag(
                            flag_id=f"flag_incomplete_med_details_{idx}",
                            category=SafetyCategory.DOSAGE_WARNING,
                            severity=SafetySeverity.LOW,
                            title=f"Incomplete Medication Regimen: {med.name}",
                            detail=(
                                f"Medication '{med.name}' is recorded with status '{med.status.value}', "
                                f"but the following regimen details are unspecified: {', '.join(missing_fields)}. "
                                "Clinician verification is required before documenting therapy or prescribing."
                            ),
                            flagged_items=[med.name],
                            requires_clinician_acknowledgment=True,
                            clinician_review_status=ReviewStatus.PENDING_REVIEW,
                            disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                        )
                    )

            # Check 2: Unconfirmed / unknown medication status
            if med.status == MedicationStatus.UNKNOWN:
                flags.append(
                    SafetyReviewFlag(
                        flag_id=f"flag_unconfirmed_med_status_{idx}",
                        category=SafetyCategory.OTHER,
                        severity=SafetySeverity.LOW,
                        title=f"Unconfirmed Medication Status: {med.name}",
                        detail=(
                            f"Medication '{med.name}' was mentioned in consultation, but its current usage "
                            "status (active, discontinued, or proposed) was not established. "
                            "Clinician clarification is recommended."
                        ),
                        flagged_items=[med.name],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )

            # Check 3: Explicitly reported non-adherence
            if med.is_adherent is False:
                flags.append(
                    SafetyReviewFlag(
                        flag_id=f"flag_med_non_adherence_{idx}",
                        category=SafetyCategory.OTHER,
                        severity=SafetySeverity.MEDIUM,
                        title=f"Reported Medication Non-Adherence: {med.name}",
                        detail=(
                            f"Patient explicitly reported non-adherence with prescribed medication '{med.name}'. "
                            "Clinician review of tolerability, cost, or dosing schedule barriers is recommended."
                        ),
                        flagged_items=[med.name],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )
        return flags

    # =================================================================
    # Rule 4: Data Quality and Contradictions
    # =================================================================

    def _check_contradictions_and_conflicts(
        self, extraction_result: ClinicalExtractionResult
    ) -> List[SafetyReviewFlag]:
        """
        Flag explicitly represented conflicting or uncertain information when the existing
        schemas provide enough evidence to identify it.
        Does not invent contradictions or assume unsupported cross-reactivities.
        """
        flags: List[SafetyReviewFlag] = []

        # 1. Drug-Allergy Conflicts (Direct match or conservative potential cross-reactivity)
        for allergy in extraction_result.allergies:
            allergen_clean = allergy.allergen.strip().lower()

            for med in extraction_result.medications:
                # Only check active, proposed, or unknown medications (ignore discontinued/denied)
                if med.status in (MedicationStatus.CURRENT, MedicationStatus.PROPOSED, MedicationStatus.UNKNOWN):
                    med_clean = med.name.strip().lower()

                    is_match, is_direct = self._is_drug_allergy_match(allergen_clean, med_clean)
                    if is_match:
                        # Determine severity based on allergy reaction severity
                        is_severe = (
                            allergy.reaction_severity in ("severe", "life-threatening", "critical")
                            or (
                                allergy.reaction
                                and any(
                                    w in allergy.reaction.lower()
                                    for w in ("anaphylaxis", "angioedema", "shock", "throat swelling", "dyspnea")
                                )
                            )
                        )
                        sev = SafetySeverity.CRITICAL if is_severe else SafetySeverity.HIGH

                        if is_direct:
                            title = f"Potential Drug-Allergy Conflict: {med.name} with Documented Allergy to {allergy.allergen}"
                            detail = (
                                f"Patient has documented allergy to '{allergy.allergen}' "
                                f"(reaction: {allergy.reaction or 'unspecified'}, "
                                f"severity: {allergy.reaction_severity or 'unspecified'}), "
                                f"and medication '{med.name}' was mentioned with status '{med.status.value}'. "
                                "Clinician verification is required before prescribing or administering."
                            )
                        else:
                            title = f"Potential Drug-Allergy Cross-Reactivity: {med.name} / {allergy.allergen}"
                            detail = (
                                f"Patient has documented allergy to '{allergy.allergen}' "
                                f"(reaction: {allergy.reaction or 'unspecified'}, "
                                f"severity: {allergy.reaction_severity or 'unspecified'}), "
                                f"which may have a potential cross-reactivity relationship with prescribed or current "
                                f"medication '{med.name}'. This potential relationship is uncertain and requires clinician "
                                "evaluation and verification; do not claim cross-reactivity is clinically established without supporting evidence."
                            )

                        flags.append(
                            SafetyReviewFlag(
                                flag_id=f"flag_allergy_conflict_{allergen_clean}_{med_clean}",
                                category=SafetyCategory.DRUG_ALLERGY_CONFLICT,
                                severity=sev,
                                title=title,
                                detail=detail,
                                flagged_items=[allergy.allergen, med.name],
                                requires_clinician_acknowledgment=True,
                                clinician_review_status=ReviewStatus.PENDING_REVIEW,
                                disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                            )
                        )

        # 2. Medication Status Contradictions (e.g. both CURRENT and DISCONTINUED/DENIED)
        med_statuses: Dict[str, Set[MedicationStatus]] = {}
        for med in extraction_result.medications:
            norm_name = med.name.strip().lower()
            med_statuses.setdefault(norm_name, set()).add(med.status)

        for norm_name, statuses in med_statuses.items():
            has_active = MedicationStatus.CURRENT in statuses or MedicationStatus.PROPOSED in statuses
            has_inactive = MedicationStatus.DISCONTINUED in statuses or MedicationStatus.DENIED in statuses
            if has_active and has_inactive:
                display_name = next(m.name for m in extraction_result.medications if m.name.strip().lower() == norm_name)
                sorted_statuses = sorted(s.value for s in statuses)
                flags.append(
                    SafetyReviewFlag(
                        flag_id=f"flag_med_status_conflict_{norm_name}",
                        category=SafetyCategory.DIAGNOSTIC_DISCREPANCY,
                        severity=SafetySeverity.MEDIUM,
                        title=f"Conflicting Medication Status: {display_name}",
                        detail=(
                            f"Medication '{display_name}' is documented with contradictory usage statuses: "
                            f"{', '.join(sorted_statuses)}. Clinician clarification is needed to confirm active medications."
                        ),
                        flagged_items=[display_name],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )

        # 3. Symptom Presence Contradictions (e.g. both PRESENT and ABSENT)
        symptom_presences: Dict[str, Set[PresenceStatus]] = {}
        for symptom in extraction_result.symptoms:
            norm_name = symptom.name.strip().lower()
            symptom_presences.setdefault(norm_name, set()).add(symptom.status)

        for norm_name, presences in symptom_presences.items():
            if PresenceStatus.PRESENT in presences and PresenceStatus.ABSENT in presences:
                display_name = next(s.name for s in extraction_result.symptoms if s.name.strip().lower() == norm_name)
                flags.append(
                    SafetyReviewFlag(
                        flag_id=f"flag_symptom_conflict_{norm_name}",
                        category=SafetyCategory.DIAGNOSTIC_DISCREPANCY,
                        severity=SafetySeverity.MEDIUM,
                        title=f"Conflicting Symptom Presence: {display_name}",
                        detail=(
                            f"Symptom '{display_name}' is documented as both present and absent/denied across "
                            "consultation entries. Clinician clarification is needed."
                        ),
                        flagged_items=[display_name],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )

        return flags

    def _is_drug_allergy_match(self, allergen: str, med_name: str) -> Tuple[bool, bool]:
        """
        Check if an allergen and medication match.

        Returns:
            Tuple[is_match, is_direct_match]:
            - is_match: True if a direct match or potential cross-reactivity relationship exists.
            - is_direct_match: True if documented names match directly (exact token or discrete word),
                               False if based on potential class-level cross-reactivity.
        """
        a_clean = allergen.strip().lower()
        m_clean = med_name.strip().lower()

        # 1. Direct name match (exact or discrete word in name)
        if a_clean == m_clean or _has_discrete_word_match(a_clean, m_clean) or _has_discrete_word_match(m_clean, a_clean):
            return True, True

        # 2. Conservative cross-reactivity relationships
        for allergen_class, related_meds in self.allergy_cross_reactivity.items():
            class_clean = allergen_class.strip().lower()
            if class_clean == a_clean or _has_discrete_word_match(class_clean, a_clean) or _has_discrete_word_match(a_clean, class_clean):
                for rel in related_meds:
                    rel_clean = rel.strip().lower()
                    if rel_clean == m_clean or _has_discrete_word_match(rel_clean, m_clean):
                        return True, False

        return False, False

    # =================================================================
    # Rule 5: Vitals and Observations
    # =================================================================

    def _check_vitals_and_observations(
        self, extraction_result: ClinicalExtractionResult
    ) -> List[SafetyReviewFlag]:
        """
        Validate only clearly supported structural or data-quality constraints on observations and vitals.
        Does not invent clinical thresholds or label a patient as having a condition.
        Does not treat missing vitals as proof of an emergency.
        """
        flags: List[SafetyReviewFlag] = []

        # 1. Vital Signs Structural Completeness Checks
        if extraction_result.vitals is not None:
            vitals = extraction_result.vitals

            # Partial blood pressure checks (structural completeness)
            if vitals.blood_pressure_systolic is not None and vitals.blood_pressure_diastolic is None:
                flags.append(
                    SafetyReviewFlag(
                        flag_id="flag_partial_bp_missing_diastolic",
                        category=SafetyCategory.MISSING_VITAL_INFO,
                        severity=SafetySeverity.LOW,
                        title="Incomplete Blood Pressure: Missing Diastolic Reading",
                        detail=(
                            f"Systolic pressure ({vitals.blood_pressure_systolic} mmHg) was recorded, but diastolic "
                            "pressure is unrecorded. Complete blood pressure documentation requires both readings."
                        ),
                        flagged_items=["blood_pressure_diastolic"],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )
            elif vitals.blood_pressure_diastolic is not None and vitals.blood_pressure_systolic is None:
                flags.append(
                    SafetyReviewFlag(
                        flag_id="flag_partial_bp_missing_systolic",
                        category=SafetyCategory.MISSING_VITAL_INFO,
                        severity=SafetySeverity.LOW,
                        title="Incomplete Blood Pressure: Missing Systolic Reading",
                        detail=(
                            f"Diastolic pressure ({vitals.blood_pressure_diastolic} mmHg) was recorded, but systolic "
                            "pressure is unrecorded. Complete blood pressure documentation requires both readings."
                        ),
                        flagged_items=["blood_pressure_systolic"],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )

        # 2. Observations Explicitly Documented as Abnormal
        for idx, obs in enumerate(extraction_result.observations, 1):
            if obs.is_normal is False:
                flags.append(
                    SafetyReviewFlag(
                        flag_id=f"flag_abnormal_observation_{idx}",
                        category=SafetyCategory.OTHER,
                        severity=SafetySeverity.MEDIUM,
                        title=f"Abnormal Clinical Observation: {obs.category.replace('_', ' ').title()}",
                        detail=(
                            f"An observation under category '{obs.category}' was explicitly documented as abnormal: "
                            f"'{obs.description}'. Requires clinician review."
                        ),
                        flagged_items=[obs.category, obs.description],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )

        return flags

    # =================================================================
    # Rule 7: Clinical Uncertainty Preservation
    # =================================================================

    def _check_uncertainties(
        self, extraction_result: ClinicalExtractionResult
    ) -> List[SafetyReviewFlag]:
        """
        Preserve explicit clinical uncertainty without converting it into confirmed findings.
        """
        flags: List[SafetyReviewFlag] = []
        for idx, symptom in enumerate(extraction_result.symptoms, 1):
            # Uncertain or suspected symptom presence
            if symptom.status == PresenceStatus.PRESENT and symptom.certainty in (
                CertaintyLevel.UNCERTAIN,
                CertaintyLevel.SUSPECTED,
            ):
                flags.append(
                    SafetyReviewFlag(
                        flag_id=f"flag_uncertain_symptom_{idx}",
                        category=SafetyCategory.OTHER,
                        severity=SafetySeverity.LOW,
                        title=f"Uncertain Clinical Finding: {symptom.name}",
                        detail=(
                            f"Symptom '{symptom.name}' is recorded with certainty level '{symptom.certainty.value}' "
                            f"(source: {symptom.source.value}). Requires clinician diagnostic evaluation to confirm or rule out."
                        ),
                        flagged_items=[symptom.name],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )

            # Indeterminate presence status
            if symptom.status == PresenceStatus.UNKNOWN:
                flags.append(
                    SafetyReviewFlag(
                        flag_id=f"flag_indeterminate_symptom_{idx}",
                        category=SafetyCategory.OTHER,
                        severity=SafetySeverity.LOW,
                        title=f"Indeterminate Symptom Status: {symptom.name}",
                        detail=(
                            f"Presence status of symptom '{symptom.name}' is unstated or ambiguous. "
                            "Inquire with patient or review record to clarify whether symptom is present or absent."
                        ),
                        flagged_items=[symptom.name],
                        requires_clinician_acknowledgment=True,
                        clinician_review_status=ReviewStatus.PENDING_REVIEW,
                        disclaimer=CLINICAL_SAFETY_DISCLAIMER,
                    )
                )
        return flags

    # =================================================================
    # Rule 6: Reference Provenance & Untrusted Data Protection
    # =================================================================

    def _attach_reference_provenance(
        self,
        flags: List[SafetyReviewFlag],
        reference_passages: List[ReferencePassage],
    ) -> List[SafetyReviewFlag]:
        """
        Attach relevant reference passages to safety review flags while preserving provenance.
        CRITICAL: Reference passage content is strictly passive text and never executed.
        """
        updated_flags: List[SafetyReviewFlag] = []

        for flag in flags:
            flag_items_lower = {item.strip().lower() for item in flag.flagged_items}
            matching_refs: List[ReferencePassage] = []

            for ref in reference_passages:
                # Passive keyword matching against title and content
                ref_text = f"{ref.title} {ref.content}".lower()
                if any(item in ref_text for item in flag_items_lower):
                    matching_refs.append(ref)

            if matching_refs:
                # Return a new SafetyReviewFlag instance with attached references
                updated_flag = flag.model_copy(update={"supporting_references": matching_refs})
                updated_flags.append(updated_flag)
            else:
                updated_flags.append(flag)

        return updated_flags

    # =================================================================
    # Finalization and Deduplication
    # =================================================================

    def _finalize_flags(self, flags: List[SafetyReviewFlag]) -> List[SafetyReviewFlag]:
        """
        Deduplicate flags and generate deterministic sequential identifiers.
        """
        seen_keys: Set[Tuple[SafetyCategory, str, Tuple[str, ...]]] = set()
        deduped: List[SafetyReviewFlag] = []

        for flag in flags:
            key = (flag.category, flag.title, tuple(sorted(flag.flagged_items)))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(flag)

        # Assign deterministic, sequential flag IDs
        finalized: List[SafetyReviewFlag] = []
        for idx, flag in enumerate(deduped, 1):
            deterministic_id = f"FLAG-{flag.category.value.upper()}-{idx:03d}"
            finalized.append(flag.model_copy(update={"flag_id": deterministic_id}))

        return finalized
