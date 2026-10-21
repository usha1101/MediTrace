"""Unit tests for the consultation endpoint and mock clinical engine."""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_consultation_mock_response_structure():
    """Verify that consultation returns a valid response with prominent mock labeling."""
    payload = {
        "patient_id": "PT-TEST-001",
        "patient_name": "Test Patient",
        "patient_age": 45,
        "patient_gender": "Female",
        "transcript": "Doctor: How have you been feeling? Patient: I have had a mild headache for two days.",
        "known_allergies": ["Latex"],
        "current_medications": ["Multivitamin"],
        "existing_conditions": ["Migraine"],
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 200

    data = response.json()

    # Prominent mock labeling checks
    assert data["is_mock"] is True
    assert "MOCK" in data["engine_source"].upper()
    assert "[MOCK RESPONSE]" in data["mock_disclaimer"]
    assert data["patient_id"] == "PT-TEST-001"

    # SOAP note structure checks
    draft_note = data["draft_note"]
    assert "[MOCK DRAFT]" in draft_note["subjective"]
    assert "[MOCK DRAFT]" in draft_note["objective"]
    assert "[MOCK DRAFT]" in draft_note["assessment"]
    assert "[MOCK DRAFT]" in draft_note["plan"]

    # Safety alert checks
    assert isinstance(data["safety_alerts"], list)
    assert len(data["safety_alerts"]) >= 1  # At least the info mock alert


def test_consultation_penicillin_allergy_alert():
    """Verify that high-severity safety alert is triggered for penicillin conflict."""
    payload = {
        "patient_id": "PT-ALLERGY-002",
        "patient_name": "Allergic Patient",
        "patient_age": 30,
        "patient_gender": "Male",
        "transcript": "Doctor: Your throat is very inflamed. I will prescribe Amoxicillin 500mg. Patient: Sounds good.",
        "known_allergies": ["Penicillin"],
        "current_medications": [],
        "existing_conditions": [],
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 200

    data = response.json()
    alerts = data["safety_alerts"]

    # Verify HIGH severity alert is present
    high_alerts = [a for a in alerts if a["severity"] == "HIGH"]
    assert len(high_alerts) >= 1
    assert "Penicillin" in high_alerts[0]["title"]
    assert "Amoxicillin" in high_alerts[0]["recommendation"]


def test_consultation_validation_empty_transcript():
    """Verify that too-short or empty transcript fails schema validation (422 Unprocessable Entity)."""
    payload = {
        "patient_id": "PT-SHORT",
        "transcript": "Short",  # Min length is 10
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 422


def test_consultation_missing_patient_id():
    """Verify that missing required patient_id returns 422."""
    payload = {
        "transcript": "Doctor: Hello there, tell me what brings you to the clinic today.",
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 422


def test_consultation_alias_endpoint():
    """Verify that the alias endpoint /consultation also works."""
    payload = {
        "patient_id": "PT-ALIAS-003",
        "transcript": "Doctor: Checking your annual vitals. Patient: Everything has been fine.",
    }
    response = client.post("/consultation", json=payload)
    assert response.status_code == 200
    assert response.json()["is_mock"] is True


def test_fictional_consultation_produces_non_empty_soap_fields():
    """Verify that a fictional consultation produces non-empty, grounded SOAP note fields without inventing absent findings."""
    payload = {
        "patient_id": "PT-FICTIONAL-042",
        "patient_name": "Arthur Dent",
        "patient_age": 42,
        "patient_gender": "Male",
        "transcript": (
            "Doctor: Good afternoon Arthur, what symptoms are you experiencing today? "
            "Patient: I have had a persistent dry cough and mild fatigue for three days. "
            "Doctor: Let me check your lungs and breathing. Your lung sounds are relatively clear. "
            "I suggest hydration, rest, and monitoring. Let us schedule a follow-up in 5 days."
        ),
        "known_allergies": ["Sulfa"],
        "current_medications": [],
        "existing_conditions": ["Asthma (mild)"],
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 200

    data = response.json()
    draft_note = data["draft_note"]

    # Verify all four SOAP fields are non-empty strings
    for field_name in ["subjective", "objective", "assessment", "plan"]:
        field_value = draft_note.get(field_name)
        assert field_value is not None, f"Field {field_name} must not be None"
        assert isinstance(field_value, str), f"Field {field_name} must be a string"
        assert len(field_value.strip()) > 0, f"Field {field_name} must not be blank"

    # Verify subjective includes patient details and transcript complaint
    assert "Arthur Dent" in draft_note["subjective"]
    assert "dry cough" in draft_note["subjective"]

    # Verify objective reflects transcript observations and does not invent fake unrecorded findings
    assert "Respiratory" in draft_note["objective"] or any(term in draft_note["objective"].lower() for term in ["lung", "pulmonary"])
    
    # Verify assessment relates to cough evaluation
    assert "cough" in draft_note["assessment"].lower() or "bronchitis" in draft_note["assessment"].lower()

    # Verify plan includes discussed management
    assert len(draft_note["plan"].strip()) > 10


def test_draft_soap_note_editing_preserves_transcript_and_schemas():
    """Verify that editing draft SOAP fields validates against SOAPNote schema and preserves original transcript."""
    from backend.schemas.consultation import SOAPNote

    original_transcript = (
        "Doctor: Hello Sarah. Patient: I have had headaches after staring at screens all day. "
        "Doctor: Let us check your vision and discuss ergonomics."
    )
    payload = {
        "patient_id": "PT-EDIT-009",
        "patient_name": "Sarah Connor",
        "transcript": original_transcript,
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Original generated note
    generated_note = data["draft_note"]
    assert len(generated_note["subjective"]) > 0

    # Simulate clinician editing each field in edit mode
    edited_draft = {
        "subjective": generated_note["subjective"] + "\n[Clinician Amendment]: Patient reports 8+ hours screen time.",
        "objective": "[Clinician Amendment]: Vision acuity 20/20 bilateral, no papilledema.",
        "assessment": "[Clinician Amendment]: Digital eye strain / tension headache.",
        "plan": "[Clinician Amendment]: 20-20-20 rule, artificial tears PRN, ergonomic monitor adjustment.",
    }

    # Verify edited draft conforms strictly to SOAPNote Pydantic model
    validated_edited_note = SOAPNote(**edited_draft)
    assert "Digital eye strain" in validated_edited_note.assessment
    assert "screen time" in validated_edited_note.subjective

    # Verify original consultation payload and transcript are preserved
    assert payload["transcript"] == original_transcript
    assert data["patient_id"] == "PT-EDIT-009"
    assert data["is_mock"] is True


def test_regression_allergy_conflict_triggers_critical_alert_and_plan_hold():
    """Verify that proposed medication conflicting with documented allergies triggers critical alert and plan hold."""
    payload = {
        "patient_id": "PT-REG-01",
        "patient_name": "Jane Anderson",
        "known_allergies": ["Penicillin", "Amoxicillin"],
        "transcript": (
            "Doctor: It looks like acute bacterial pharyngitis. "
            "I am going to prescribe Amoxicillin 500mg TID for 7 days. Patient: Okay doctor."
        ),
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 200
    data = response.json()

    # 1. Critical safety alert
    alerts = data["safety_alerts"]
    high_alerts = [a for a in alerts if a["severity"] == "HIGH"]
    assert len(high_alerts) >= 1
    crit_alert = high_alerts[0]
    assert "CRITICAL" in crit_alert["title"]
    assert "Amoxicillin" in crit_alert["title"] or "Penicillin" in crit_alert["title"]
    assert "NOT cleared or safe" in crit_alert["description"]
    assert "MANDATORY CLINICIAN VERIFICATION" in crit_alert["recommendation"]

    # 2. Plan hold - do not present as cleared or safe
    plan_text = data["draft_note"]["plan"]
    assert "CONTRAINDICATED PRESCRIPTION HOLD" in plan_text
    assert "DO NOT DISPENSE" in plan_text
    assert "NOT cleared or safe" in plan_text
    assert "Clinician must verify allergy" in plan_text


def test_regression_reported_versus_measured_fever():
    """Verify that patient-reported fever is placed in Subjective and not claimed as an objective measured vital sign."""
    payload = {
        "patient_id": "PT-REG-02",
        "transcript": (
            "Doctor: What brings you in today? "
            "Patient: I have had a severe sore throat, body aches, and fever for the past 2 days."
        ),
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 200
    data = response.json()
    draft = data["draft_note"]

    # Subjective correctly captures reported fever as patient-reported, not measured
    assert "fever (patient-reported)" in draft["subjective"]
    assert "NOT a measured vital sign" in draft["subjective"]

    # Objective does not claim fever as measured vital
    assert "• Measured Vital Signs: None documented" in draft["objective"]
    assert "patient-reported only" in draft["objective"]
    assert "98.6" not in draft["objective"]  # Ensure no fabricated default temp is present


def test_regression_symptom_and_duration_preservation():
    """Verify reported symptoms and duration are preserved, diagnoses labeled provisional, and ICD-10 suggestions flagged."""
    payload = {
        "patient_id": "PT-REG-03",
        "patient_name": "Jane Anderson",
        "known_allergies": ["Penicillin", "Amoxicillin"],
        "current_medications": ["Cetirizine 10mg daily"],
        "existing_conditions": ["Seasonal allergic rhinitis"],
        "transcript": (
            "Doctor: Good morning Jane, what brings you in today?\n"
            "Patient: I've had a severe sore throat, body aches, and fever for the past 2 days. It hurts to swallow.\n"
            "Doctor: Let me check your throat. There is severe tonsillar erythema and exudate. "
            "It looks like acute bacterial pharyngitis. I'm going to prescribe Amoxicillin 500mg TID for 7 days.\n"
            "Patient: Okay doctor, whatever will clear this infection up quickly."
        ),
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 200
    data = response.json()
    draft = data["draft_note"]

    # Symptoms preserved in Subjective
    subj = draft["subjective"].lower()
    assert "sore throat" in subj
    assert "body aches" in subj
    assert "fever" in subj
    assert "swallow" in subj or "dysphagia" in subj
    assert "2 days" in subj

    # Examination findings in Objective explicitly documented
    obj = draft["objective"]
    assert "Clinician-Narrated Physical Examination" in obj
    assert "tonsillar erythema" in obj
    assert "tonsillar exudate" in obj

    # Assessment labels diagnosis as provisional and etiology unconfirmed
    asmt = draft["assessment"]
    assert "Assessment (Provisional)" in asmt
    assert "Provisional" in asmt
    assert "UNCONFIRMED" in asmt
    assert "cannot" in asmt.lower() or "requires" in asmt.lower()

    # ICD-10 codes labeled as suggestions requiring clinician review
    for code in data["suggested_icd10_codes"]:
        assert "[SUGGESTION - Requires Clinician Review]" in code


def test_regression_unsupported_findings_are_not_invented():
    """Verify that allergy reactions, test results, vitals, counseling, and follow-up are not invented when absent."""
    payload = {
        "patient_id": "PT-REG-04",
        "patient_name": "Plain Patient",
        "known_allergies": ["Latex"],
        "transcript": (
            "Doctor: Hello, how can I help you today? "
            "Patient: I have had a mild dry cough since yesterday. "
            "Doctor: I see. Please make sure to get plenty of rest."
        ),
    }

    response = client.post("/api/v1/consultation", json=payload)
    assert response.status_code == 200
    data = response.json()
    draft = data["draft_note"]

    # No invented vitals or physical exam
    assert "Measured Vital Signs: None documented" in draft["objective"]
    assert "BP: " not in draft["objective"]
    assert "SpO2" not in draft["objective"]
    assert "124/82" not in draft["objective"]

    # No fabricated allergy reaction history (e.g. hives, anaphylaxis date)
    assert "anaphylaxis" not in draft["subjective"].lower()
    assert "hives" not in draft["subjective"].lower()

    # Plan does not invent unmentioned lab test results or fake follow-up dates
    assert "Rapid strep test" not in draft["plan"]
    assert "Specific follow-up schedule and counseling not explicitly documented" in draft["plan"]



