"""MediTrace - Ambient Clinical Copilot with Real-Time Safety Verification.

Streamlit Frontend User Interface (Phase 6 Integration).
Communicates with FastAPI backend; displays structured draft SOAP notes,
real-time safety verification flags, missing clinical information,
guideline reference provenance, and explicit mock vs live status.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

# Ensure repository root is on sys.path for robust imports across environments
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontend.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiHttpError,
    ApiPayloadError,
    ApiTimeoutError,
    check_backend_health,
    extract_missing_information,
    extract_references,
    extract_safety_flags,
    extract_soap_fields,
    get_default_backend_url,
    get_retrieval_status_info,
    is_mock_response,
    is_mock_transcription,
    submit_consultation,
    submit_transcription,
)

# Configure Streamlit page
st.set_page_config(
    page_title="MediTrace - Ambient Clinical Copilot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical dashboard styling
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
        margin-bottom: 1.2rem;
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
    .live-badge {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #6EE7B7;
        display: inline-block;
    }
    .status-badge {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
        border: 1px solid #C7D2FE;
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
    .ref-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .missing-card {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .disclaimer-card {
        background-color: #F3F4F6;
        border: 1px solid #D1D5DB;
        border-left: 4px solid #6B7280;
        border-radius: 6px;
        padding: 0.85rem;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 0.75rem;
    }
    @media (prefers-color-scheme: dark) {
        .soap-card {
            background-color: #1F2937;
            color: #F9FAFB;
            border-color: #374151;
        }
        .ref-card {
            background-color: #1E293B;
            border-color: #334155;
            color: #F8FAFC;
        }
        .missing-card {
            background-color: #292524;
            border-color: #78350F;
            color: #FEF3C7;
        }
        .disclaimer-card {
            background-color: #1F2937;
            border-color: #4B5563;
            color: #E5E7EB;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# SIDEBAR CONFIGURATION
# =====================================================================
st.sidebar.image(
    "https://img.icons8.com/fluency/96/stethoscope.png",
    width=64,
)
st.sidebar.title("MediTrace Controls")

# Configurable backend base URL with environment variable resolution
default_url = get_default_backend_url()
backend_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value=default_url,
    help="Configurable via MEDITRACE_BACKEND_URL or FASTAPI_BACKEND_URL environment variables.",
)

# Backend Health Check
health_info = check_backend_health(backend_url)

if health_info["online"]:
    mock_mode = health_info.get("mock_engine_active", False)
    mock_text = "Active (Scaffold)" if mock_mode else "Inactive (Live LangGraph Engine)"
    st.sidebar.success(
        f"**Backend Online** (v{health_info.get('version', '0.1.0')})\n\n"
        f"• **AI Engine:** {mock_text}\n\n"
        f"• **Service:** `{health_info.get('service', 'meditrace-backend')}`"
    )
else:
    st.sidebar.error(
        f"**Backend Offline**\n\n"
        f"Could not connect to `{backend_url}/health`.\n\n"
        f"Reason: {health_info.get('error', 'Connection refused')}\n\n"
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

# Clinical Scenarios
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

# Session state management for preset selection
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
    "• **Frontend:** Streamlit Clinical Dashboard (Member B)\n"
    "• **API Gateway:** FastAPI Service with async routes\n"
    "• **Workflow:** LangGraph StateGraph connecting Member A Clinical AI\n"
    "• **Speech Engine:** Local On-Device Transcription\n\n"
    "*(Draft documentation and safety alerts require qualified clinician validation. "
    "Absence of an alert does not prove patient safety.)*"
)


# =====================================================================
# AUDIO TRANSCRIPTION HELPER
# =====================================================================
def request_transcription(audio_bytes: bytes, filename: str) -> None:
    """Submit recorded or uploaded audio to the FastAPI transcription endpoint."""
    if not health_info["online"]:
        st.error(
            f"Cannot transcribe audio: FastAPI backend is unreachable at `{backend_url}`. "
            "Please ensure the backend is running before transcribing audio."
        )
        return

    with st.spinner("Transcribing encounter audio via speech-to-text service..."):
        try:
            data = submit_transcription(
                audio_bytes=audio_bytes,
                filename=filename,
                base_url=backend_url,
            )
            st.session_state["transcript_content"] = data.get("transcript", "")
            st.session_state["stt_info"] = data
            engine_name = data.get("engine_name", "STT")
            if data.get("is_mock"):
                st.warning(f"Audio processed using mock speech engine `{engine_name}`.")
            else:
                st.success(f"Transcribed successfully using on-device engine `{engine_name}`!")
            st.rerun()

        except ApiTimeoutError as exc:
            st.error(f"Speech transcription timed out: {exc}")
        except ApiConnectionError as exc:
            st.error(f"Could not connect to transcription service at {backend_url}: {exc}")
        except ApiHttpError as exc:
            st.error(f"Transcription failed ({exc}): {exc.detail}")
        except ApiPayloadError as exc:
            st.error(f"Malformed transcription response: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error during transcription: {str(exc)}")


# =====================================================================
# MAIN HEADER & SYSTEM STATUS BANNER
# =====================================================================
st.markdown('<div class="main-header">🩺 MediTrace Clinical Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ambient Clinical Documentation with Real-Time Safety Verification</div>',
    unsafe_allow_html=True,
)

# Prominent Notice regarding system engine status
if health_info["online"]:
    if health_info.get("mock_engine_active", False):
        st.info(
            "**ℹ️ PROTOTYPE ENGINE NOTICE:** Backend is currently running with **MockClinicalEngine** "
            "(Member B integration scaffold). Draft notes and safety flags are simulated for testing "
            "until live Gemini credentials are provided.",
            icon="⚠️",
        )
    else:
        st.success(
            "**🔒 LIVE CLINICAL WORKFLOW ACTIVE:** Backend connected to **LangGraph Clinical Workflow** "
            "(Gemini 2.5 Flash + Qdrant Evidence Retrieval + Deterministic Safety Engine). "
            "All generated outputs require licensed clinician review.",
            icon="🩺",
        )
else:
    st.error(
        f"**🚨 BACKEND UNAVAILABLE:** The MediTrace FastAPI backend at `{backend_url}` is offline. "
        "Please start the backend server to process clinical encounters.",
        icon="🔌",
    )


# =====================================================================
# SECTION 1: PATIENT CONTEXT & DEMOGRAPHICS
# =====================================================================
st.subheader("1. Patient Context & History")

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


# =====================================================================
# SECTION 2: AMBIENT AUDIO INGESTION (SPEECH-TO-TEXT)
# =====================================================================
st.subheader("2. Ambient Audio Ingestion (Speech-to-Text)")

audio_tab1, audio_tab2, audio_tab3 = st.tabs(
    ["🎤 Live Microphone Recording", "📁 Upload Audio File", "📝 Manual / Preset Dialogue"]
)

with audio_tab1:
    st.caption("Record ambient clinical encounter dialogue using your microphone.")
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

# Display STT provenance badge if transcription was run
if "stt_info" in st.session_state:
    stt_meta = st.session_state["stt_info"]
    if stt_meta.get("is_mock"):
        st.warning(
            f"🎙️ **Transcription Mode:** `{stt_meta.get('engine_name')}` (Simulated Audio Transcript). "
            "Please review and verify the transcribed narrative below prior to clinical analysis.",
            icon="⚠️",
        )
    else:
        st.success(
            f"🎙️ **On-Device Speech Recognition:** `{stt_meta.get('engine_name')}` | "
            f"Language: `{stt_meta.get('language', 'en')}` | "
            f"Duration: `{stt_meta.get('duration_seconds', 0)}s` | "
            "Local execution (no audio leaves client/server hardware).",
            icon="🔒",
        )


# =====================================================================
# SECTION 3: CLINICIAN REVIEW & TRANSCRIPT EDITING
# =====================================================================
st.subheader("3. Clinician Review & Transcript Editing")
st.caption("Review, edit, or append to the encounter transcript before generating the clinical note.")

transcript_input = st.text_area(
    "Doctor-Patient Conversation Transcript *",
    value=st.session_state.get("transcript_content", ""),
    height=170,
    placeholder="Doctor: Hello, how can I help you today? \nPatient: I've been experiencing...",
    help="Ambient clinical dialogue captured during patient visit or transcribed from audio.",
)

# Maintain session state synchronization
st.session_state["transcript_content"] = transcript_input

col_btn1, col_btn2 = st.columns([4, 6])
with col_btn1:
    submit_btn = st.button("🚀 Generate SOAP Note & Verify Safety", type="primary", use_container_width=True)


# =====================================================================
# SUBMISSION LOGIC: INTEGRATING TEXT CONSULTATIONS
# =====================================================================
if submit_btn:
    if not patient_id.strip():
        st.error("Patient ID is required.")
    elif not transcript_input.strip() or len(transcript_input.strip()) < 10:
        st.error("Please provide a valid conversation transcript (at least 10 non-whitespace characters).")
    elif not health_info["online"]:
        st.error(
            f"Cannot connect to MediTrace backend at `{backend_url}`. "
            "Please ensure the FastAPI server is running before submitting."
        )
    else:
        # Prepare list fields
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

        with st.spinner("Processing clinical encounter through MediTrace workflow..."):
            try:
                response_data = submit_consultation(
                    payload=payload,
                    base_url=backend_url,
                )
                st.session_state["consultation_result"] = response_data
                st.session_state["soap_edit_mode"] = False
                st.session_state["working_soap_note"] = dict(response_data.get("draft_note", {}))
                st.success("Consultation processed successfully!")

            except ApiConnectionError as exc:
                st.error(
                    f"🔌 **Backend Connection Error:** Could not connect to `{backend_url}`. "
                    f"Verify that the FastAPI backend is running. ({exc})"
                )
            except ApiTimeoutError as exc:
                st.error(
                    f"⏱️ **Request Timeout:** The clinical AI workflow took longer than expected to respond. ({exc})"
                )
            except ApiHttpError as exc:
                st.error(
                    f"⚠️ **Backend Error ({exc.status_code}):** {exc.detail}"
                )
            except ApiPayloadError as exc:
                st.error(
                    f"📦 **Malformed Response:** Backend returned unparseable or invalid payload: {exc}"
                )
            except ApiClientError as exc:
                st.error(f"❌ **Validation Error:** {exc}")
            except Exception as exc:
                st.error(f"❌ **Unexpected Error:** {str(exc)}")


# =====================================================================
# SECTION 4: DISPLAY CONSULTATION ANALYSIS & DOCUMENTATION
# =====================================================================
if "consultation_result" in st.session_state:
    result = st.session_state["consultation_result"]

    st.markdown("---")
    st.subheader("4. Consultation Analysis & Safety Verification")

    # Metadata & Provenance Header
    is_mock = is_mock_response(result)
    engine_source = result.get("engine_source", "Unknown Engine")
    consultation_id = result.get("consultation_id", "N/A")
    encounter_id = result.get("encounter_id") or "N/A"
    created_at = result.get("created_at", "")[:19].replace("T", " ")

    col_meta1, col_meta2 = st.columns([7, 3])
    with col_meta1:
        if is_mock:
            badge_html = f'<span class="mock-badge">⚠️ MOCK ENGINE OUTPUT: {engine_source}</span>'
        else:
            badge_html = f'<span class="live-badge">✅ LIVE WORKFLOW: {engine_source}</span>'

        st.markdown(
            f"{badge_html} "
            f'<span style="color: #6B7280; font-size: 0.85rem; margin-left: 8px;">'
            f'Consultation ID: <code>{consultation_id}</code> | '
            f'Encounter ID: <code>{encounter_id}</code> | '
            f'Timestamp: {created_at} UTC</span>',
            unsafe_allow_html=True,
        )
    with col_meta2:
        workflow_status = result.get("workflow_status", "completed").upper()
        retrieval_status_val = result.get("retrieval_status")
        retrieval_info = get_retrieval_status_info(retrieval_status_val)
        st.markdown(
            f'<div style="text-align: right;">'
            f'<span class="status-badge">Status: {workflow_status}</span> '
            f'<span class="status-badge">Retrieval: {retrieval_info["label"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Show mock disclaimer banner if mock response
    if is_mock:
        st.warning(
            f"**⚠️ Mock Output Notice:** {result.get('mock_disclaimer', 'Output generated by mock scaffold.')}",
            icon="⚠️",
        )

    # Organize outputs into clear, distinct tabs
    tab_soap, tab_safety, tab_missing, tab_references, tab_disclaimer, tab_raw = st.tabs(
        [
            "📋 Draft SOAP Note",
            "🛡️ Real-Time Safety Verification",
            "❓ Missing Information",
            "📚 Clinical References",
            "📜 Safety Disclaimer",
            "🔍 Raw API Response",
        ]
    )

    # -----------------------------------------------------------------
    # TAB 1: DRAFT SOAP NOTE
    # -----------------------------------------------------------------
    with tab_soap:
        soap_fields = extract_soap_fields(result)
        note_status = soap_fields.get("status", "draft_pending_review").upper()

        st.markdown(
            f'<div style="margin-bottom: 0.75rem;">'
            f'<strong>Lifecycle Status:</strong> <span class="status-badge">{note_status}</span> '
            f'<span style="color: #6B7280; font-size: 0.85rem; margin-left: 10px;">'
            f'Note ID: <code>{soap_fields.get("note_id") or "PENDING"}</code> | '
            f'Human-in-the-loop clinician review is mandatory prior to signing.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Initialize working_soap_note if not present
        if "working_soap_note" not in st.session_state:
            st.session_state["working_soap_note"] = dict(result.get("draft_note") or {})

        is_edit_mode = st.session_state.get("soap_edit_mode", False)

        if is_edit_mode:
            st.info(
                "✏️ **Edit Mode Active:** You are modifying the working draft note below. "
                "Saving updates your working draft for clinician review, but does **not** finalize or sign the note.",
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
                if st.button("💾 Save Working Draft", type="primary", use_container_width=True):
                    st.session_state["working_soap_note"] = {
                        "subjective": edit_sub,
                        "objective": edit_obj,
                        "assessment": edit_asmt,
                        "plan": edit_plan,
                    }
                    result["draft_note"] = st.session_state["working_soap_note"]
                    st.session_state["soap_edit_mode"] = False
                    st.toast("Draft note updated. Pending clinician approval.", icon="💾")
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
                st.markdown("**Suggested ICD-10 Diagnostic Codes (Suggestions Only):**")
                st.write(", ".join([f"`{code}`" for code in result["suggested_icd10_codes"]]))

            st.markdown("---")
            action_col1, action_col2 = st.columns([3, 7])
            with action_col1:
                if st.button("✏️ Edit Working Draft", type="secondary", use_container_width=True):
                    st.session_state["soap_edit_mode"] = True
                    st.rerun()

            with action_col2:
                if st.button("✅ Sign & Finalize Note (Prototype)", use_container_width=True):
                    st.warning(
                        "⚠️ **Prototype Notice:** Autonomous clinical note finalization, prescribing, and EHR export "
                        "are strictly disabled in this educational prototype. Final clinical sign-off must occur "
                        "through an authorized Electronic Health Record (EHR) system by a licensed physician.",
                        icon="ℹ️",
                    )

    # -----------------------------------------------------------------
    # TAB 2: REAL-TIME SAFETY VERIFICATION
    # -----------------------------------------------------------------
    with tab_safety:
        safety_flags = extract_safety_flags(result)

        if not safety_flags:
            st.info(
                "ℹ️ **No Safety Flags Triggered:** No explicit drug-allergy, contraindication, or dosage alerts "
                "were identified for this encounter text."
            )
        else:
            st.markdown(f"**Identified Clinical Safety Flags ({len(safety_flags)}):**")
            for flag in safety_flags:
                sev = flag.get("severity", "INFO").upper()
                if sev in ("HIGH", "CRITICAL"):
                    css_class = "alert-high"
                    icon = "🚨"
                elif sev == "MEDIUM":
                    css_class = "alert-medium"
                    icon = "⚠️"
                else:
                    css_class = "alert-info"
                    icon = "ℹ️"

                flagged_items = flag.get("flagged_items") or []
                items_str = f" | Flagged: {', '.join(flagged_items)}" if flagged_items else ""

                st.markdown(
                    f"""
                    <div class="{css_class}">
                        <strong>{icon} [{sev}] {flag.get('title')}</strong> ({flag.get('category')}{items_str})<br>
                        <p style="margin: 0.4rem 0 0.4rem 0;">{flag.get('detail')}</p>
                        <small><strong>Recommended Action:</strong> {flag.get('recommendation')}</small><br>
                        <small style="color: #6B7280;">Review Status: <code>{flag.get('review_status')}</code></small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Critical Clinical Safety Notice: Never infer safety from absence of alert
        st.markdown(
            """
            <div class="disclaimer-card">
                <strong>⚠️ Clinical Safety Notice:</strong>
                The absence of a safety alert or contraindication flag must <strong>NEVER</strong> be interpreted
                as proof that a patient or proposed treatment is safe. MediTrace is an educational clinical copilot;
                all prescriptions, dosages, and diagnostic decisions must be independently reviewed and verified
                by a licensed healthcare practitioner.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------
    # TAB 3: MISSING CLINICAL INFORMATION
    # -----------------------------------------------------------------
    with tab_missing:
        missing_items = extract_missing_information(result)

        if not missing_items:
            st.info(
                "ℹ️ **No Explicit Documentation Gaps Identified:** The consultation dialogue provided "
                "standard baseline context, or no omission categories met the alerting threshold."
            )
        else:
            st.markdown(
                f"**Clinical Information Gaps Identified ({len(missing_items)}):**\n"
                "*The following clinical details were not discussed or recorded during the encounter.*"
            )
            for item in missing_items:
                cat = item.get("category", "General Clinical Data")
                desc = item.get("description", "No description provided.")
                rel = item.get("clinical_relevance", "Clinician review recommended.")
                follow_up = item.get("recommended_follow_up")

                follow_up_html = (
                    f"<br><small><strong>Suggested Follow-up:</strong> {follow_up}</small>"
                    if follow_up
                    else ""
                )

                st.markdown(
                    f"""
                    <div class="missing-card">
                        <strong>🔍 [{cat}]</strong> {desc}<br>
                        <p style="margin: 0.35rem 0 0.2rem 0; font-size: 0.95rem;">
                            <strong>Clinical Relevance:</strong> {rel}
                        </p>
                        {follow_up_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption(
            "Preventing omission errors: Reviewing missing items helps ensure vital signs, allergy checks, "
            "and medication histories are not overlooked prior to patient discharge."
        )

    # -----------------------------------------------------------------
    # TAB 4: CLINICAL REFERENCES & GUIDELINE PROVENANCE
    # -----------------------------------------------------------------
    with tab_references:
        retrieval_status_val = result.get("retrieval_status")
        retrieval_info = get_retrieval_status_info(retrieval_status_val)

        col_ret1, col_ret2 = st.columns([6, 4])
        with col_ret1:
            st.markdown(
                f"**Guideline Retrieval Status:** <span class=\"status-badge\">{retrieval_info['label']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(retrieval_info["description"])

        references = extract_references(result)

        if not references:
            st.info(
                "ℹ️ No reference passages were retrieved or attached for this clinical encounter."
            )
        else:
            st.markdown(f"**Retrieved Evidence Passages ({len(references)}):**")
            for ref in references:
                source_doc = ref.get("source_document", "Clinical Guideline")
                title = ref.get("title", "Reference Passage")
                section = ref.get("section_or_page")
                section_str = f" | Section: {section}" if section else ""
                score = ref.get("relevance_score")
                score_str = f" | Relevance: {score:.1%}" if score is not None else ""
                url_doi = ref.get("source_url_or_doi")
                url_html = (
                    f'<br><small><a href="{url_doi}" target="_blank">🔗 Source Link / DOI</a></small>'
                    if url_doi
                    else ""
                )

                st.markdown(
                    f"""
                    <div class="ref-card">
                        <strong>📖 {source_doc}</strong> — <em>{title}</em>{section_str}{score_str}<br>
                        <p style="margin: 0.4rem 0 0.3rem 0; font-size: 0.92rem; font-style: italic;">
                            "{ref.get('content')}"
                        </p>
                        <small style="color: #6B7280;">Passage ID: <code>{ref.get('passage_id')}</code></small>
                        {url_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------
    # TAB 5: CLINICAL SAFETY DISCLAIMER & AUDIT
    # -----------------------------------------------------------------
    with tab_disclaimer:
        clinical_disclaimer = (
            result.get("clinical_disclaimer")
            or (result.get("draft_note") or {}).get("disclaimer")
            or "EDUCATIONAL PROTOTYPE ONLY: MediTrace does not autonomously diagnose or prescribe."
        )

        st.markdown("#### Clinical Safety Disclaimer")
        st.markdown(
            f"""
            <div class="disclaimer-card" style="border-left-color: #EF4444;">
                <strong>⚠️ MANDATORY REGULATORY NOTICE:</strong><br>
                {clinical_disclaimer}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Audit & Encounter Provenance")
        st.markdown(
            f"""
            - **Consultation ID:** `{consultation_id}`
            - **Encounter ID:** `{encounter_id}`
            - **Patient ID:** `{result.get('patient_id')}`
            - **Engine Source:** `{engine_source}`
            - **Workflow Status:** `{result.get('workflow_status', 'completed')}`
            - **Mock Engine Active:** `{is_mock}`
            - **Guideline Retrieval Status:** `{retrieval_status_val or 'not_reported'}`
            - **Generated Timestamp:** `{created_at} UTC`
            """
        )

    # -----------------------------------------------------------------
    # TAB 6: RAW API RESPONSE
    # -----------------------------------------------------------------
    with tab_raw:
        st.caption("Inspect raw JSON payload received from the MediTrace FastAPI backend.")
        st.json(result)
