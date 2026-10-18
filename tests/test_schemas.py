"""
Unit tests for MediTrace Clinical AI Engine schemas (app/schemas.py).
Tests valid data, invalid data, missing/uncertain information preservation,
and clinical validation behavior.
"""

import pytest
from pydantic import ValidationError

from app.schemas import (
    CLINICAL_SAFETY_DISCLAIMER,
    AllergyInfo,
    CertaintyLevel,
    ClinicalExtractionResult,
    DraftSOAPNote,
    InformationSource,
    MedicationMention,
    MedicationStatus,
    MissingInformation,
    NoteStatus,
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


# =====================================================================
# 1. SymptomInfo Tests
# =====================================================================

def test_symptom_info_valid_default():
    symptom = SymptomInfo(name="Shortness of breath")
    assert symptom.name == "Shortness of breath"
    assert symptom.status == PresenceStatus.PRESENT
    # Default certainty must be UNCERTAIN: does not imply confirmation from report alone
    assert symptom.certainty == CertaintyLevel.UNCERTAIN
    assert symptom.source == InformationSource.PATIENT_REPORTED
    assert symptom.body_site is None
    assert symptom.raw_quote is None


def test_symptom_info_preserves_uncertainty_and_negation():
    # Patient denies chest pain
    symptom = SymptomInfo(
        name="Chest pain",
        status=PresenceStatus.ABSENT,
        certainty=CertaintyLevel.DENIED,
        source=InformationSource.PATIENT_REPORTED,
        raw_quote="Patient stated: 'I don't have any chest pain.'",
    )
    assert symptom.status == PresenceStatus.ABSENT
    assert symptom.certainty == CertaintyLevel.DENIED
    assert "chest pain" in symptom.raw_quote.lower()


def test_symptom_info_rejects_empty_name():
    with pytest.raises(ValidationError) as exc_info:
        SymptomInfo(name="   ")
    assert "Symptom name must not be empty or whitespace only" in str(exc_info.value)


def test_symptom_info_rejects_confirmed_certainty_with_unknown_presence():
    with pytest.raises(ValidationError) as exc_info:
        SymptomInfo(
            name="Dizziness",
            status=PresenceStatus.UNKNOWN,
            certainty=CertaintyLevel.CONFIRMED,
        )
    assert "Cannot set certainty to CONFIRMED when presence status is UNKNOWN" in str(exc_info.value)


# =====================================================================
# 2. AllergyInfo Tests
# =====================================================================

def test_allergy_info_valid_unverified_default():
    allergy = AllergyInfo(
        allergen="Penicillin",
        reaction="Hives and facial swelling",
        raw_quote="I broke out in hives when I took penicillin as a child",
    )
    assert allergy.allergen == "Penicillin"
    assert allergy.source == InformationSource.PATIENT_REPORTED
    assert allergy.verified_by_clinician is False
    assert allergy.reaction == "Hives and facial swelling"


def test_allergy_info_verified_by_clinician():
    allergy = AllergyInfo(
        allergen="Sulfa drugs",
        reaction="Anaphylaxis",
        source=InformationSource.INDEPENDENTLY_VERIFIED,
        verified_by_clinician=True,
    )
    assert allergy.verified_by_clinician is True
    assert allergy.source == InformationSource.INDEPENDENTLY_VERIFIED


def test_allergy_info_preserves_unknown_reaction():
    # Allergy stated but reaction is unknown/unstated
    allergy = AllergyInfo(allergen="Aspirin")
    assert allergy.allergen == "Aspirin"
    assert allergy.reaction is None
    assert allergy.reaction_severity is None


def test_allergy_info_rejects_blank_allergen():
    with pytest.raises(ValidationError) as exc_info:
        AllergyInfo(allergen="")
    assert "Allergen must not be empty or whitespace only" in str(exc_info.value)


def test_allergy_info_rejects_contradictory_patient_reported_and_verified():
    with pytest.raises(ValidationError) as exc_info:
        AllergyInfo(
            allergen="Penicillin",
            source=InformationSource.PATIENT_REPORTED,
            verified_by_clinician=True,
        )
    assert "source cannot be 'patient_reported' when verified_by_clinician is True" in str(exc_info.value)


def test_allergy_info_rejects_contradictory_verified_source_with_false_flag():
    with pytest.raises(ValidationError) as exc_info:
        AllergyInfo(
            allergen="Penicillin",
            source=InformationSource.INDEPENDENTLY_VERIFIED,
            verified_by_clinician=False,
        )
    assert "source is 'independently_verified' but verified_by_clinician is False" in str(exc_info.value)


def test_allergy_info_rejects_contradictory_unknown_source_with_verified():
    with pytest.raises(ValidationError) as exc_info:
        AllergyInfo(
            allergen="Penicillin",
            source=InformationSource.UNKNOWN,
            verified_by_clinician=True,
        )
    assert "source cannot be 'unknown' when verified_by_clinician is True" in str(exc_info.value)


# =====================================================================
# 3. MedicationMention Tests
# =====================================================================

def test_medication_mention_valid_full():
    med = MedicationMention(
        name="Lisinopril",
        dosage="10 mg",
        route="oral",
        frequency="daily",
        status=MedicationStatus.CURRENT,
        source=InformationSource.PATIENT_REPORTED,
        is_adherent=True,
        raw_quote="I take 10 mg lisinopril every morning",
    )
    assert med.name == "Lisinopril"
    assert med.dosage == "10 mg"
    assert med.status == MedicationStatus.CURRENT
    assert med.is_adherent is True


def test_medication_mention_default_status_is_unknown():
    # Mentioning a medication does not establish patient currently takes it
    med = MedicationMention(name="Metformin")
    assert med.name == "Metformin"
    assert med.status == MedicationStatus.UNKNOWN


def test_unspecified_medication_fields_remain_none():
    # Unspecified attributes must remain None and never be hallucinated or filled with defaults
    med = MedicationMention(name="Amoxicillin")
    assert med.dosage is None
    assert med.route is None
    assert med.frequency is None
    assert med.duration is None
    assert med.is_adherent is None
    assert med.raw_quote is None


def test_medication_mention_rejects_blank_name():
    with pytest.raises(ValidationError) as exc_info:
        MedicationMention(name="  \t  ")
    assert "Medication name must not be empty or whitespace only" in str(exc_info.value)


# =====================================================================
# 4. VitalSigns & ObservationInfo Tests
# =====================================================================

def test_vital_signs_valid():
    vitals = VitalSigns(
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        heart_rate=72,
        respiratory_rate=16,
        temperature_celsius=36.8,
        oxygen_saturation_pct=98.0,
        source=InformationSource.CLINICIAN_ENTERED,
    )
    assert vitals.blood_pressure_systolic == 120
    assert vitals.blood_pressure_diastolic == 80
    assert vitals.oxygen_saturation_pct == 98.0


def test_unspecified_vital_signs_remain_none():
    # Unmeasured vital signs must remain None and not default to synthetic numbers
    vitals = VitalSigns()
    assert vitals.blood_pressure_systolic is None
    assert vitals.blood_pressure_diastolic is None
    assert vitals.heart_rate is None
    assert vitals.respiratory_rate is None
    assert vitals.temperature_celsius is None
    assert vitals.oxygen_saturation_pct is None
    assert vitals.measurement_time is None
    assert vitals.notes is None


def test_partial_vital_signs_preserves_unspecified_as_none():
    vitals = VitalSigns(heart_rate=78)
    assert vitals.heart_rate == 78
    assert vitals.blood_pressure_systolic is None
    assert vitals.blood_pressure_diastolic is None
    assert vitals.respiratory_rate is None
    assert vitals.temperature_celsius is None
    assert vitals.oxygen_saturation_pct is None


def test_vital_signs_rejects_systolic_less_than_or_equal_to_diastolic():
    # Systolic cannot be lower than diastolic
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(blood_pressure_systolic=80, blood_pressure_diastolic=120)
    assert "Systolic pressure (80 mmHg) must be strictly greater than diastolic" in str(exc_info.value)

    # Equal systolic and diastolic is also physiologically invalid
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(blood_pressure_systolic=90, blood_pressure_diastolic=90)
    assert "Systolic pressure (90 mmHg) must be strictly greater than diastolic" in str(exc_info.value)


def test_vital_signs_rejects_unusual_values_with_clear_error_messages():
    # Systolic BP high
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(blood_pressure_systolic=350)
    assert "Systolic blood pressure (350 mmHg) is outside physiologically plausible range (40-300 mmHg)" in str(exc_info.value)

    # Systolic BP low
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(blood_pressure_systolic=30)
    assert "Systolic blood pressure (30 mmHg) is outside physiologically plausible range (40-300 mmHg)" in str(exc_info.value)

    # Diastolic BP high
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(blood_pressure_diastolic=220)
    assert "Diastolic blood pressure (220 mmHg) is outside physiologically plausible range (20-200 mmHg)" in str(exc_info.value)

    # Diastolic BP low
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(blood_pressure_diastolic=10)
    assert "Diastolic blood pressure (10 mmHg) is outside physiologically plausible range (20-200 mmHg)" in str(exc_info.value)

    # Heart rate high
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(heart_rate=350)
    assert "Heart rate (350 bpm) is outside physiologically plausible range (20-300 bpm)" in str(exc_info.value)

    # Heart rate low
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(heart_rate=15)
    assert "Heart rate (15 bpm) is outside physiologically plausible range (20-300 bpm)" in str(exc_info.value)

    # Respiratory rate high
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(respiratory_rate=95)
    assert "Respiratory rate (95 breaths/min) is outside physiologically plausible range (4-80 breaths/min)" in str(exc_info.value)

    # Respiratory rate low
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(respiratory_rate=2)
    assert "Respiratory rate (2 breaths/min) is outside physiologically plausible range (4-80 breaths/min)" in str(exc_info.value)

    # Temperature high
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(temperature_celsius=48.0)
    assert "Body temperature (48.0 °C) is outside physiologically plausible range (28.0-45.0 °C)" in str(exc_info.value)

    # Temperature low
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(temperature_celsius=25.0)
    assert "Body temperature (25.0 °C) is outside physiologically plausible range (28.0-45.0 °C)" in str(exc_info.value)

    # SpO2 high
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(oxygen_saturation_pct=105.0)
    assert "Oxygen saturation (105.0%) is outside physiologically plausible range (40.0-100.0%)" in str(exc_info.value)

    # SpO2 low
    with pytest.raises(ValidationError) as exc_info:
        VitalSigns(oxygen_saturation_pct=30.0)
    assert "Oxygen saturation (30.0%) is outside physiologically plausible range (40.0-100.0%)" in str(exc_info.value)


def test_observation_info_valid():
    obs = ObservationInfo(
        category="physical_exam",
        description="Lungs clear to auscultation bilaterally. No wheezes or crackles.",
        is_normal=True,
        source=InformationSource.CLINICIAN_ENTERED,
    )
    assert obs.category == "physical_exam"
    assert obs.is_normal is True


def test_observation_info_rejects_blank_fields():
    with pytest.raises(ValidationError):
        ObservationInfo(category="", description="Normal exam")

    with pytest.raises(ValidationError):
        ObservationInfo(category="exam", description="   ")


# =====================================================================
# 5. MissingInformation Tests
# =====================================================================

def test_missing_information_valid():
    missing = MissingInformation(
        category="medication_dosage",
        description="Patient stated taking Atorvastatin but omitted dose and frequency.",
        clinical_relevance="Dose needed to evaluate lipid target compliance and drug interactions.",
        recommended_follow_up="Check recent pharmacy fill or query patient regarding tablet strength.",
    )
    assert missing.category == "medication_dosage"
    assert "Atorvastatin" in missing.description
    assert missing.recommended_follow_up is not None


def test_missing_information_rejects_empty_fields():
    with pytest.raises(ValidationError):
        MissingInformation(
            category="",
            description="Missing allergy record",
            clinical_relevance="Safety check",
        )


# =====================================================================
# 7. ReferencePassage Tests
# =====================================================================

def test_reference_passage_valid():
    ref = ReferencePassage(
        passage_id="ref_nice_ng28_01",
        title="Type 2 diabetes in adults: management (NG28)",
        content="First-line pharmacotherapy is metformin unless contraindicated.",
        source_document="NICE Guideline NG28",
        relevance_score=0.92,
    )
    assert ref.passage_id == "ref_nice_ng28_01"
    assert ref.relevance_score == 0.92


def test_reference_passage_rejects_invalid_score():
    with pytest.raises(ValidationError):
        ReferencePassage(
            passage_id="ref_01",
            title="Guideline",
            content="Content",
            source_document="Source",
            relevance_score=1.5,  # Exceeds 1.0 limit
        )


# =====================================================================
# 8. SafetyReviewFlag Tests
# =====================================================================

def test_safety_review_flag_valid():
    flag = SafetyReviewFlag(
        flag_id="flag_allergy_001",
        category=SafetyCategory.DRUG_ALLERGY_CONFLICT,
        severity=SafetySeverity.HIGH,
        title="Potential Beta-Lactam Cross-Reactivity Risk",
        detail="Patient reported severe hives with Penicillin; Amoxicillin prescribed in draft plan.",
        flagged_items=["Penicillin", "Amoxicillin"],
    )
    assert flag.flag_id == "flag_allergy_001"
    assert flag.severity == SafetySeverity.HIGH
    assert flag.requires_clinician_acknowledgment is True
    assert flag.clinician_review_status == ReviewStatus.PENDING_REVIEW
    assert CLINICAL_SAFETY_DISCLAIMER in flag.disclaimer


def test_safety_review_flag_rejects_blank_disclaimer():
    with pytest.raises(ValidationError) as exc_info:
        SafetyReviewFlag(
            flag_id="flag_002",
            category=SafetyCategory.OTHER,
            severity=SafetySeverity.LOW,
            title="Warning",
            detail="Detail",
            disclaimer="   ",
        )
    assert "disclaimer must not be cleared or empty" in str(exc_info.value)


# =====================================================================
# 6. ClinicalExtractionResult Tests
# =====================================================================

def test_clinical_extraction_result_composition():
    symptom = SymptomInfo(name="Productive cough")
    allergy = AllergyInfo(allergen="Cephalosporins")
    med = MedicationMention(name="Albuterol inhaler")
    vitals = VitalSigns(heart_rate=80, oxygen_saturation_pct=97.0)
    missing = MissingInformation(
        category="vitals",
        description="Temperature not documented",
        clinical_relevance="Assessment of fever in respiratory infection",
    )

    result = ClinicalExtractionResult(
        encounter_id="enc_12345",
        symptoms=[symptom],
        allergies=[allergy],
        medications=[med],
        vitals=vitals,
        missing_information=[missing],
    )

    assert result.encounter_id == "enc_12345"
    assert len(result.symptoms) == 1
    assert result.symptoms[0].name == "Productive cough"
    assert len(result.allergies) == 1
    assert result.vitals.oxygen_saturation_pct == 97.0
    assert len(result.missing_information) == 1
    assert CLINICAL_SAFETY_DISCLAIMER in result.disclaimer


# =====================================================================
# 9. DraftSOAPNote Tests
# =====================================================================

def test_draft_soap_note_valid():
    soap = DraftSOAPNote(
        subjective="Patient reports 3-day history of sore throat and fatigue. Denies fever.",
        objective="Pharynx erythematous without exudate. Vitals: HR 74, BP 118/76, Temp 37.0 C.",
        assessment="Acute pharyngitis, likely viral etiology. Review required.",
        plan="1. Supportive care and hydration. 2. Clinician signoff required for symptomatic therapy.",
    )
    assert soap.status == NoteStatus.DRAFT_PENDING_REVIEW
    assert "Acute pharyngitis" in soap.assessment
    assert CLINICAL_SAFETY_DISCLAIMER in soap.disclaimer


def test_draft_soap_note_rejects_blank_section():
    with pytest.raises(ValidationError) as exc_info:
        DraftSOAPNote(
            subjective="",  # Invalid blank section
            objective="Vitals normal",
            assessment="Viral URI",
            plan="Rest",
        )
    assert "SOAP section 'subjective' must not be empty or whitespace only" in str(exc_info.value)


def test_draft_soap_note_with_attached_flags_and_references():
    flag = SafetyReviewFlag(
        flag_id="flag_003",
        category=SafetyCategory.MISSING_VITAL_INFO,
        severity=SafetySeverity.MEDIUM,
        title="Missing Oxygen Saturation",
        detail="Patient reports shortness of breath, but SpO2 was not recorded.",
    )
    ref = ReferencePassage(
        passage_id="ref_002",
        title="Asthma exacerbation guideline",
        content="Assess SpO2 in all patients presenting with acute dyspnea.",
        source_document="BTS/SIGN British Guideline",
    )

    soap = DraftSOAPNote(
        subjective="Shortness of breath on exertion.",
        objective="Lungs: bilateral expiratory wheezes.",
        assessment="Mild asthma exacerbation.",
        plan="Albuterol 2.5 mg nebulizer; observe response.",
        safety_flags=[flag],
        references=[ref],
    )
    assert len(soap.safety_flags) == 1
    assert soap.safety_flags[0].category == SafetyCategory.MISSING_VITAL_INFO
    assert len(soap.references) == 1
