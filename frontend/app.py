"""MediTrace - Ambient Clinical Copilot with Real-Time Safety Verification.

Streamlit Frontend User Interface (Member B - App Integration).
Communicates with FastAPI backend; clearly labels mock engine responses.
"""

import json
from datetime import datetime
import requests
import streamlit as st

# Configure page
st.set_page_config(
    page_title="MediTrace - Ambient Clinical Copilot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical dashboard look
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .mock-badge {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #FCD34D;
        display: inline-block;
    }
    .alert-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 0.85rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    .alert-medium {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.85rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    .alert-info {
        background-color: #E0F2FE;
        border-left: 5px solid #0284C7;
        padding: 0.85rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
    }
    .soap-card {
        background-color: #F9FAFB;
        color: #111827;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        white-space: pre-wrap;
        font-family: inherit;
        line-height: 1.5;
    }
    @media (prefers-color-scheme: dark) {
        .soap-card {
            background-color: #1F2937;
            color: #F9FAFB;
            border-color: #374151;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- SIDEBAR CONFIGURATION ---
st.sidebar.image(
    "https://img.icons8.com/fluency/96/stethoscope.png",
    width=64,
)
st.sidebar.title("MediTrace Controls")
backend_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value="http://127.0.0.1:8000",
    help="Base URL of the MediTrace FastAPI service",
)

# Backend Health Check Helper
def check_backend_health(url: str) -> dict:
    try:
        resp = requests.get(f"{url.rstrip('/')}/health", timeout=2.5)
        if resp.status_code == 200:
            return {"online": True, "data": resp.json()}
        return {"online": False, "error": f"Status code: {resp.status_code}"}
    except Exception as e:
        return {"online": False, "error": str(e)}

health_info = check_backend_health(backend_url)

if health_info["online"]:
    st.sidebar.success(
        f"Backend Online (v{health_info['data'].get('version', '0.1.0')})\n\n"
        f"Mock Engine: {health_info['data'].get('mock_engine_active', True)}"
    )
else:
    st.sidebar.error(
        f"Backend Offline!\n\nCould not connect to {backend_url}/health.\n\n"
        "Start backend with:\n`uvicorn backend.main:app --reload`"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Test Presets")
preset = st.sidebar.selectbox(
    "Load Clinical Scenario",
    [
        "Scenario 1: Penicillin Allergy Conflict (High Risk)",
        "Scenario 2: Routine Follow-up (Normal)",
        "Custom Blank Scenario",
    ],
)

# Preset data dictionary
PRESETS = {
    "Scenario 1: Penicillin Allergy Conflict (High Risk)": {
        "patient_id": "PT-9402",
        "name": "Jane Anderson",
        "age": 34,
        "gender": "Female",
        "allergies": "Penicillin, Amoxicillin",
        "meds": "Cetirizine 10mg daily",
        "conditions": "Seasonal allergic rhinitis",
        "transcript": (
            "Doctor: Good morning Jane, what brings you in today?\n"
            "Patient: I've had a severe sore throat, body aches, and fever for the past 2 days. It hurts to swallow.\n"
            "Doctor: Let me check your throat. There is severe tonsillar erythema and exudate. "
            "It looks like acute bacterial pharyngitis. I'm going to prescribe Amoxicillin 500mg TID for 7 days.\n"
            "Patient: Okay doctor, whatever will clear this infection up quickly."
        ),
    },
    "Scenario 2: Routine Follow-up (Normal)": {
        "patient_id": "PT-8105",
        "name": "Robert Miller",
        "age": 62,
        "gender": "Male",
        "allergies": "No known drug allergies",
        "meds": "Lisinopril 20mg daily, Metformin 500mg BID",
        "conditions": "Type 2 Diabetes Mellitus, Essential Hypertension",
        "transcript": (
            "Doctor: Good morning Robert, how are your blood pressure and blood sugar readings lately?\n"
            "Patient: Both have been very stable, doctor. My fasting glucose is usually around 110 and BP 124/80.\n"
            "Doctor: Your lab results from last week confirm an HbA1c of 6.8%. "
            "Your vitals today look very reassuring. We will maintain your Lisinopril and Metformin dosages as is.\n"
            "Patient: Sounds great, thank you doctor."
        ),
    },
    "Custom Blank Scenario": {
        "patient_id": "",
        "name": "",
        "age": 45,
        "gender": "Other",
        "allergies": "",
        "meds": "",
        "conditions": "",
        "transcript": "",
    },
}

selected_data = PRESETS[preset]

# Track transcript content in session state for review & edit workflow
if "last_selected_preset" not in st.session_state:
    st.session_state["last_selected_preset"] = preset
    st.session_state["transcript_content"] = selected_data["transcript"]

if st.session_state["last_selected_preset"] != preset:
    st.session_state["last_selected_preset"] = preset
    st.session_state["transcript_content"] = selected_data["transcript"]
    if "stt_info" in st.session_state:
        del st.session_state["stt_info"]

st.sidebar.markdown("---")
st.sidebar.caption(
    "**MediTrace Integration Architecture**\n\n"
    "• Member B: App integration, Streamlit UI, FastAPI service, Speech-to-Text\n"
    "• Member A: Clinical AI engine & real-time verification model\n\n"
    "*Clinical analysis responses are labeled mock prototypes until Member A's engine is connected.*"
)

# Helper function to submit audio to backend transcription
def request_transcription(audio_bytes: bytes, filename: str):
    if not health_info["online"]:
        st.error(f"Cannot transcribe audio: backend is offline at {backend_url}.")
        return
    with st.spinner("Transcribing clinical audio on-device..."):
        try:
            files = {"file": (filename, audio_bytes, "application/octet-stream")}
            resp = requests.post(
                f"{backend_url.rstrip('/')}/api/v1/transcribe",
                files=files,
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["transcript_content"] = data["transcript"]
                st.session_state["stt_info"] = data
                st.success(f"Transcribed successfully using {data.get('engine_name', 'STT')}!")
                st.rerun()
            else:
                st.error(f"Transcription failed (HTTP {resp.status_code}): {resp.text}")
        except Exception as exc:
            st.error(f"Failed to connect to transcription service: {str(exc)}")

# --- MAIN CONTENT AREA ---
st.markdown('<div class="main-header">🩺 MediTrace Clinical Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ambient Clinical Documentation & Real-Time Safety Verification</div>',
    unsafe_allow_html=True,
)

# Prominent Mock Banner for Clinical AI
st.info(
    "**ℹ️ PROTOTYPE INTEGRATION NOTICE:** Operating with **MockClinicalEngine** "
    "(Member B integration scaffold). Draft SOAP notes and safety alerts are simulated for end-to-end "
    "workflow testing until Member A's AI engine is plugged in. Speech-to-Text runs locally on-device.",
    icon="⚠️",
)

# Patient Input Section
st.subheader("1. Patient Context")

with st.expander("Patient Demographics & Medical History", expanded=True):
    col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
    with col1:
        patient_id = st.text_input("Patient ID *", value=selected_data["patient_id"] or "PT-1001")
    with col2:
        patient_name = st.text_input("Full Name", value=selected_data["name"])
    with col3:
        patient_age = st.number_input("Age", min_value=0, max_value=125, value=selected_data["age"] or 40)
    with col4:
        gender_options = ["Female", "Male", "Non-binary", "Other", "Prefer not to say"]
        gender_idx = (
            gender_options.index(selected_data["gender"])
            if selected_data["gender"] in gender_options
            else 0
        )
        patient_gender = st.selectbox("Gender", gender_options, index=gender_idx)

    col5, col6, col7 = st.columns(3)
    with col5:
        known_allergies_str = st.text_input(
            "Known Allergies (comma-separated)",
            value=selected_data["allergies"],
            help="e.g. Penicillin, Sulfa, Latex",
        )
    with col6:
        current_meds_str = st.text_input(
            "Current Medications (comma-separated)",
            value=selected_data["meds"],
            help="e.g. Lisinopril 10mg, Metformin 500mg",
        )
    with col7:
        existing_conditions_str = st.text_input(
            "Existing Conditions (comma-separated)",
            value=selected_data["conditions"],
            help="e.g. Hypertension, Asthma",
        )

# Audio Ingestion Section
st.subheader("2. Ambient Audio Ingestion (Speech-to-Text)")

audio_tab1, audio_tab2, audio_tab3 = st.tabs(
    ["🎤 Live Microphone Recording", "📁 Upload Audio File", "📝 Manual / Preset Dialogue"]
)

with audio_tab1:
    st.caption("Record ambient clinical encounter dialogue using your system microphone.")
    if hasattr(st, "audio_input"):
        mic_audio = st.audio_input("Record Doctor-Patient Encounter", key="mic_recorder")
        if mic_audio is not None:
            st.audio(mic_audio)
            if st.button("🔄 Transcribe Recorded Audio", key="btn_transcribe_mic"):
                audio_bytes = mic_audio.getvalue()
                request_transcription(audio_bytes, "microphone_recording.wav")
    else:
        st.info("Direct microphone capture is not supported in this Streamlit version. Use Audio File Upload.")

with audio_tab2:
    st.caption("Upload recorded clinical encounter audio (.wav, .mp3, .m4a, .ogg, .webm, .flac).")
    uploaded_file = st.file_uploader(
        "Choose an encounter audio file",
        type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
        key="file_uploader",
    )
    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("🔄 Transcribe Uploaded Audio", key="btn_transcribe_upload"):
            audio_bytes = uploaded_file.getvalue()
            request_transcription(audio_bytes, uploaded_file.name)

with audio_tab3:
    st.caption("Direct text input is active. Select a scenario from the sidebar or type dialogue below.")

# Display STT Source & Privacy Badge if transcription was executed
if "stt_info" in st.session_state:
    stt_meta = st.session_state["stt_info"]
    if stt_meta.get("is_mock"):
        st.warning(
            f"🎙️ **Transcription Mode:** `{stt_meta.get('engine_name')}` (Simulated Audio Transcript). "
            "Please review and edit the transcribed narrative below prior to clinical analysis.",
            icon="⚠️",
        )
    else:
        st.success(
            f"🎙️ **Local On-Device Transcription:** `{stt_meta.get('engine_name')}` | "
            f"Language: `{stt_meta.get('language', 'en')}` | "
            f"Duration: `{stt_meta.get('duration_seconds', 0)}s` | "
            "Private & HIPAA-friendly (runs entirely on local hardware).",
            icon="🔒",
        )

# Transcript Review & Edit Section
st.subheader("3. Clinician Review & Transcript Editing")
st.caption("Review, edit, or append to the encounter transcript before generating the clinical note.")

transcript_input = st.text_area(
    "Doctor-Patient Conversation Transcript *",
    value=st.session_state.get("transcript_content", ""),
    height=170,
    placeholder="Doctor: Hello, how can I help you today? \nPatient: I've been experiencing...",
    help="Ambient clinical dialogue captured during patient visit or transcribed from audio.",
)

# Keep session state updated with clinician manual edits:
st.session_state["transcript_content"] = transcript_input

col_btn1, col_btn2 = st.columns([4, 6])
with col_btn1:
    submit_btn = st.button("🚀 Generate SOAP Note & Verify Safety", type="primary", use_container_width=True)


# Consultation Submission Logic
if submit_btn:
    if not patient_id.strip():
        st.error("Patient ID is required.")
    elif not transcript_input.strip() or len(transcript_input.strip()) < 10:
        st.error("Please provide a valid conversation transcript (at least 10 characters).")
    elif not health_info["online"]:
        st.error(
            f"Cannot connect to backend at {backend_url}. Please ensure the FastAPI server is running."
        )
    else:
        # Prepare lists
        allergies_list = [a.strip() for a in known_allergies_str.split(",") if a.strip()]
        meds_list = [m.strip() for m in current_meds_str.split(",") if m.strip()]
        conditions_list = [c.strip() for c in existing_conditions_str.split(",") if c.strip()]

        payload = {
            "patient_id": patient_id.strip(),
            "patient_name": patient_name.strip() if patient_name else None,
            "patient_age": int(patient_age),
            "patient_gender": patient_gender,
            "transcript": transcript_input.strip(),
            "known_allergies": allergies_list,
            "current_medications": meds_list,
            "existing_conditions": conditions_list,
        }

        with st.spinner("Processing clinical encounter with MediTrace backend..."):
            try:
                resp = requests.post(
                    f"{backend_url.rstrip('/')}/api/v1/consultation",
                    json=payload,
                    timeout=10,
                )
                if resp.status_code == 200:
                    st.session_state["consultation_result"] = resp.json()
                    st.session_state["soap_edit_mode"] = False
                    st.session_state["working_soap_note"] = dict(resp.json().get("draft_note", {}))
                    st.success("Consultation processed successfully!")
                else:
                    st.error(f"Backend returned error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")

# Display Results if available
if "consultation_result" in st.session_state:
    result = st.session_state["consultation_result"]

    st.markdown("---")
    st.subheader("2. Consultation Analysis & Documentation")

    # Mock Engine Banner Tag
    c_left, c_right = st.columns([6, 4])
    with c_left:
        st.markdown(
            f'<span class="mock-badge">⚠️ {result.get("engine_source", "MOCK ENGINE")}</span> '
            f'<span style="color: #6B7280; font-size: 0.85rem; margin-left: 8px;">'
            f'Consultation ID: <code>{result.get("consultation_id")}</code> | '
            f'Time: {result.get("created_at")[:19].replace("T", " ")} UTC</span>',
            unsafe_allow_html=True,
        )
    with c_right:
        st.caption(f"Mock Flag: `{result.get('is_mock', True)}` (Member A Engine: Ready for Hookup)")

    tab_soap, tab_safety, tab_raw = st.tabs(
        ["📋 Draft SOAP Note", "🛡️ Real-Time Safety Verification", "🔍 Raw API Response"]
    )

    with tab_soap:
        # Initialize working_soap_note if not present
        if "working_soap_note" not in st.session_state:
            st.session_state["working_soap_note"] = dict(result.get("draft_note") or {})

        is_edit_mode = st.session_state.get("soap_edit_mode", False)

        if is_edit_mode:
            st.info(
                "✏️ **Edit Mode Active:** You are editing the working draft note below. "
                "Saving updates your draft note for review, but does **not** sign or finalize the clinical note.",
                icon="📝",
            )
            current_note = st.session_state["working_soap_note"]

            edit_sub = st.text_area(
                "Subjective (S) - Patient History & Narrative",
                value=current_note.get("subjective", ""),
                height=130,
                key="edit_soap_subjective",
                help="Patient-reported symptoms, allergies, and history.",
            )
            edit_obj = st.text_area(
                "Objective (O) - Vital Signs & Physical Examination",
                value=current_note.get("objective", ""),
                height=130,
                key="edit_soap_objective",
                help="Observed vital signs, laboratory data, and physical examination findings.",
            )
            edit_asmt = st.text_area(
                "Assessment (A) - Clinical Diagnoses & Reasoning",
                value=current_note.get("assessment", ""),
                height=130,
                key="edit_soap_assessment",
                help="Primary clinical diagnosis and differential diagnoses.",
            )
            edit_plan = st.text_area(
                "Plan (P) - Treatment, Prescriptions & Follow-up",
                value=current_note.get("plan", ""),
                height=130,
                key="edit_soap_plan",
                help="Therapeutic interventions, medications, patient education, and follow-up.",
            )

            col_save, col_cancel = st.columns([3, 3])
            with col_save:
                if st.button("💾 Save Draft Changes", type="primary", use_container_width=True):
                    st.session_state["working_soap_note"] = {
                        "subjective": edit_sub,
                        "objective": edit_obj,
                        "assessment": edit_asmt,
                        "plan": edit_plan,
                    }
                    # Keep result['draft_note'] synchronized for exports / raw inspector
                    result["draft_note"] = st.session_state["working_soap_note"]
                    st.session_state["soap_edit_mode"] = False
                    st.toast("Draft changes saved. Note is not finalized or signed.", icon="💾")
                    st.rerun()

            with col_cancel:
                if st.button("✖️ Cancel Editing", use_container_width=True):
                    st.session_state["soap_edit_mode"] = False
                    st.rerun()

        else:
            note = st.session_state["working_soap_note"]
            st.markdown("#### Subjective (S)")
            subjective_text = note.get("subjective") or "[No subjective narrative generated]"
            st.info(subjective_text)

            st.markdown("#### Objective (O)")
            objective_text = note.get("objective") or "[No objective findings recorded]"
            st.markdown(f'<div class="soap-card">{objective_text}</div>', unsafe_allow_html=True)

            st.markdown("#### Assessment (A)")
            assessment_text = note.get("assessment") or "[No clinical assessment generated]"
            st.markdown(f'<div class="soap-card">{assessment_text}</div>', unsafe_allow_html=True)

            st.markdown("#### Plan (P)")
            plan_text = note.get("plan") or "[No plan generated]"
            st.markdown(f'<div class="soap-card">{plan_text}</div>', unsafe_allow_html=True)

            if result.get("suggested_icd10_codes"):
                st.markdown("**Suggested ICD-10 Diagnostic Codes:**")
                st.write(", ".join([f"`{code}`" for code in result["suggested_icd10_codes"]]))

            st.markdown("---")
            action_col1, action_col2, action_col3 = st.columns([3, 3, 4])
            with action_col1:
                if st.button("✏️ Edit Note", type="secondary", use_container_width=True):
                    st.session_state["soap_edit_mode"] = True
                    st.rerun()
            with action_col2:
                if st.button("✅ Sign & Finalize Note", use_container_width=True):
                    st.toast("Clinical note signed by clinician.", icon="✅")


    with tab_safety:
        alerts = result.get("safety_alerts", [])
        if not alerts:
            st.success("No clinical safety contraindications flagged for this encounter.")
        else:
            for alert in alerts:
                sev = alert.get("severity", "INFO").upper()
                if sev == "HIGH":
                    css_class = "alert-high"
                    icon = "🚨"
                elif sev == "MEDIUM":
                    css_class = "alert-medium"
                    icon = "⚠️"
                else:
                    css_class = "alert-info"
                    icon = "ℹ️"

                st.markdown(
                    f"""
                    <div class="{css_class}">
                        <strong>{icon} [{sev}] {alert.get('title')}</strong> ({alert.get('category')})<br>
                        <p style="margin: 0.35rem 0 0.35rem 0;">{alert.get('description')}</p>
                        <small><strong>Recommendation:</strong> {alert.get('recommendation')}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab_raw:
        st.json(result)
