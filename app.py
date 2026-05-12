"""
app.py - AI Pharmacy Assistant
Main Streamlit application.

Tab 1 — Prescription Reader + AI Pharmacy Chatbot
Tab 2 — Clinical Pharmacology RAG Chatbot (answers from the textbook)

Run with:
    streamlit run app.py
"""

import html

import pandas as pd
import streamlit as st
from PIL import Image

from modules.chatbot import (
    get_chatbot_response,
    get_rag_response,
)
from modules.extractor import analyze_prescription, get_medicines_list
from modules.rag import load_default_pdf


st.set_page_config(
    page_title="AI Pharmacy Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    .main { background-color: #f0f4f8; }

    /* ---- Header ---- */
    .app-header {
        background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .app-header h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
    .app-header p  { font-size: 1rem; margin: 0.4rem 0 0; opacity: 0.9; }

    /* ---- Section titles ---- */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a73e8;
        border-left: 4px solid #1a73e8;
        padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem;
    }

    /* ---- Medicine card ---- */
    .medicine-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1a73e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .medicine-card h3 {
        color: #1a73e8;
        margin: 0 0 0.6rem;
        font-size: 1.1rem;
    }
    .medicine-card .detail-row {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        font-size: 0.9rem;
        color: #444;
    }
    .medicine-card .detail-item strong { color: #222; }

    /* ---- Confidence badges ---- */
    .badge-high   { background:#d4edda; color:#155724; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-medium { background:#fff3cd; color:#856404; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-low    { background:#f8d7da; color:#721c24; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }

    /* ---- Info / warning / rag boxes ---- */
    .info-box {
        background: #e8f0fe; border: 1px solid #c5d8fb;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.88rem; color: #1a3a6b; margin-bottom: 1rem;
    }
    .warning-box {
        background: #fff8e1; border: 1px solid #ffe082;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.88rem; color: #5d4037; margin-bottom: 1rem;
    }
    .rag-box {
        background: #f3e5f5; border: 1px solid #ce93d8;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.88rem; color: #4a148c; margin-bottom: 1rem;
    }

    /* ---- Sidebar ---- */
    .sidebar-info {
        background: #e8f0fe; border-radius: 8px;
        padding: 0.8rem; font-size: 0.85rem; color: #1a3a6b;
    }

    footer { visibility: hidden; }
    /* ---- Chat messages ---- */
    .stChatMessage { font-size: 1.05rem !important; line-height: 1.6; }
    .stChatMessage p { margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)



_TAB1_DEFAULTS = {
    "extraction_result": None,
    "medicines":         [],
    "chat_messages":     [],
    "uploaded_image":    None,
    "prescription_meta": {},
}

_TAB2_DEFAULTS = {
    "rag_index":      None,
    "rag_chunks":     [],
    "rag_messages":   [],
    "rag_pdf_stats":  {},
    "rag_load_error": None,
}

for key, default in {**_TAB1_DEFAULTS, **_TAB2_DEFAULTS}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.rag_index is None and st.session_state.rag_load_error is None:
    _pdf_result = load_default_pdf()
    if "error" in _pdf_result:
        st.session_state.rag_load_error = _pdf_result["error"]
    else:
        st.session_state.rag_index     = _pdf_result["index"]
        st.session_state.rag_chunks    = _pdf_result["chunks"]
        st.session_state.rag_pdf_stats = {
            "pages":  _pdf_result["total_pages"],
            "chunks": _pdf_result["total_chunks"],
        }


import os

try:
    GOOGLE_API_KEYS = list(st.secrets.get("GOOGLE_API_KEYS", []))
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except FileNotFoundError:
    google_env = os.environ.get("GOOGLE_API_KEYS", "")
    GOOGLE_API_KEYS = [k.strip() for k in google_env.split(",") if k.strip()]
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

with st.sidebar:
    st.markdown("### 📋 About")
    st.markdown("""
    <div class="sidebar-info">
    <b>AI Pharmacy Assistant</b><br><br>
    🖼️ <b>Tab 1:</b> Upload prescription → extract medicines → chatbot<br><br>
    📚 <b>Tab 2:</b> Ask questions answered from the Clinical Pharmacology textbook
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🔄 Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


st.markdown("""
<div class="app-header">
    <h1>💊 AI Pharmacy Assistant</h1>
    <p>Prescription Reader &nbsp;·&nbsp; Medicine Chatbot &nbsp;·&nbsp; Clinical Pharmacology RAG</p>
</div>
""", unsafe_allow_html=True)


tab1, tab2 = st.tabs([
    "🖼️ Prescription & Medicine Chat",
    "📚 Clinical Pharmacology RAG Chat",
])


with tab1:

    col_upload, col_info = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-title">📤 Upload Prescription</div>',
                    unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a prescription image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of your prescription (handwritten or printed)",
            label_visibility="collapsed",
            key="prescription_uploader",
        )

        if uploaded_file is not None:
            if uploaded_file.type not in {"image/jpeg", "image/png", "image/jpg"}:
                st.error("❌ Unsupported file type. Please upload a JPG or PNG image.")
            else:
                try:
                    st.session_state.uploaded_image = Image.open(uploaded_file)
                except Exception as e:
                    st.error(f"❌ Could not open image: {e}")
                    st.session_state.uploaded_image = None

        if st.session_state.uploaded_image is not None:
            st.image(
                st.session_state.uploaded_image,
                caption="📄 Uploaded Prescription",
                use_container_width=True,
            )

            if st.button("🔍 Analyze Prescription", type="primary",
                         use_container_width=True, key="analyze_btn"):
                with st.spinner("🔬 Analyzing prescription with AI Vision..."):
                    result = analyze_prescription(
                        st.session_state.uploaded_image,
                        GOOGLE_API_KEYS,
                        groq_api_key=GROQ_API_KEY,
                    )

                if "error" in result:
                    st.error(f"❌ {result['error']}")
                    st.session_state.extraction_result = None
                    st.session_state.medicines = []
                else:
                    st.session_state.extraction_result = result
                    st.session_state.medicines = get_medicines_list(result)
                    st.session_state.prescription_meta = {
                        "doctor_name":        result.get("doctor_name"),
                        "patient_name":       result.get("patient_name"),
                        "date":               result.get("date"),
                        "prescription_notes": result.get("prescription_notes"),
                    }
                    st.session_state.chat_messages = []
                    st.success(
                        f"✅ Found {len(st.session_state.medicines)} medicine(s)!"
                    )
                    st.rerun()
        else:
            st.markdown("""
            <div class="info-box">
            📌 <b>How to use:</b><br>
            1. Upload a prescription image (JPG or PNG)<br>
            2. Click <b>Analyze Prescription</b><br>
            3. View extracted medicines<br>
            4. Ask the chatbot any questions
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-title">📋 Prescription Details</div>',
                    unsafe_allow_html=True)

        meta = st.session_state.prescription_meta
        if meta:
            patient = html.escape(meta.get("patient_name") or "—")
            doctor  = html.escape(meta.get("doctor_name") or "—")
            date    = html.escape(meta.get("date") or "—")
            notes   = html.escape(meta.get("prescription_notes") or "—")
            
            st.markdown(f"""
            <div style="background: white; border-radius: 10px; padding: 1.2rem; border-top: 4px solid #1a73e8; box-shadow: 0 2px 8px rgba(0,0,0,0.07);">
                <div style="margin-bottom: 0.8rem;"><b>👤 Patient Name:</b> <span style="color: #444;">{patient}</span></div>
                <div style="margin-bottom: 0.8rem;"><b>🩺 Doctor Name:</b> <span style="color: #444;">{doctor}</span></div>
                <div style="margin-bottom: 0.8rem;"><b>📅 Date:</b> <span style="color: #444;">{date}</span></div>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px dashed #ddd;">
                    <b>📝 Prescription Notes & Instructions:</b><br>
                    <span style="color: #444; display: inline-block; margin-top: 0.4rem;">{notes}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
            Prescription details will appear here after you analyze an image.
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">💊 Extracted Medicines</div>',
                unsafe_allow_html=True)

    if st.session_state.medicines:
        view_mode = st.radio(
            "Display mode", ["🃏 Cards", "📊 Table"],
            horizontal=True, label_visibility="collapsed",
        )

        if view_mode == "🃏 Cards":
            card_cols = st.columns(2)
            for idx, med in enumerate(st.session_state.medicines):
                with card_cols[idx % 2]:
                    confidence  = (med.get("confidence") or "medium").lower()
                    name     = html.escape(med.get("medicine_name") or "Unknown Medicine")
                    dosage   = html.escape(med.get("dosage")        or "—")
                    freq     = html.escape(med.get("frequency")     or "—")
                    duration = html.escape(med.get("duration")      or "—")
                    route    = html.escape(med.get("route")         or "—")
                    st.markdown(f"""
                    <div class="medicine-card">
                        <h3>💊 {name}</h3>
                        <div class="detail-row">
                            <div class="detail-item"><strong>Dosage:</strong><br>{dosage}</div>
                            <div class="detail-item"><strong>Frequency:</strong><br>{freq}</div>
                            <div class="detail-item"><strong>Duration:</strong><br>{duration}</div>
                            <div class="detail-item"><strong>Route:</strong><br>{route}</div>
                        </div>
                        <div style="margin-top:0.6rem;">
                            Confidence:&nbsp;
                            <span class="badge-{confidence}">{confidence.capitalize()}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.dataframe(
                pd.DataFrame([{
                    "Medicine":   m.get("medicine_name") or "—",
                    "Dosage":     m.get("dosage")        or "—",
                    "Frequency":  m.get("frequency")     or "—",
                    "Duration":   m.get("duration")      or "—",
                    "Route":      m.get("route")         or "—",
                    "Confidence": (m.get("confidence") or "medium").capitalize(),
                } for m in st.session_state.medicines]),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Important:</b> Always verify extracted information against your original
        prescription. Consult your pharmacist if anything looks incorrect.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-box">
        💊 Extracted medicines will appear here after you analyze a prescription.
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">🤖 AI Pharmacy Chatbot</div>',
                unsafe_allow_html=True)

    if not st.session_state.medicines:
        st.markdown("""
        <div class="info-box">
        💬 The chatbot will be available after you analyze a prescription.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
        💬 Ask me anything about your prescribed medicines — usage, side effects,
        interactions, storage, and more.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**💡 Suggested questions:**")
        _rx_suggestions = [
            "What are the most common and severe side effects?",
            "Are there any food, drink, or lifestyle restrictions?",
            "Could these medications interact with each other?",
            "What should I do if I accidentally miss a dose?",
            "How exactly do these medicines work in my body?",
        ]
        for i, sug in enumerate(st.columns(len(_rx_suggestions))):
            if sug.button(_rx_suggestions[i], key=f"rx_sug_{i}",
                          use_container_width=True):
                _q = _rx_suggestions[i]
                st.session_state.chat_messages.append(
                    {"role": "user", "content": _q}
                )
                with st.spinner("Thinking..."):
                    _resp = get_chatbot_response(
                        user_message=_q,
                        chat_history=st.session_state.chat_messages[:-1],
                        medicines=st.session_state.medicines,
                        groq_api_key=GROQ_API_KEY,
                    )
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": _resp}
                )
                st.rerun()

        for _msg in st.session_state.chat_messages:
            with st.chat_message(
                _msg["role"],
                avatar="🧑" if _msg["role"] == "user" else "💊",
            ):
                st.markdown(_msg["content"])

        _user_input = st.chat_input(
            "Ask about your medicines (e.g. 'What are the side effects of Amoxicillin?')",
            key="rx_chat_input",
        )
        if _user_input:
            st.session_state.chat_messages.append(
                {"role": "user", "content": _user_input}
            )
            with st.chat_message("user", avatar="🧑"):
                st.markdown(_user_input)

            with st.chat_message("assistant", avatar="💊"):
                with st.spinner("Thinking..."):
                    _response = get_chatbot_response(
                        user_message=_user_input,
                        chat_history=st.session_state.chat_messages[:-1],
                        medicines=st.session_state.medicines,
                        groq_api_key=GROQ_API_KEY,
                    )
                st.markdown(_response)

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": _response}
            )
            st.rerun()

        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat", key="clear_rx_chat"):
                st.session_state.chat_messages = []
                st.rerun()


with tab2:

    st.markdown('<div class="section-title">📚 Clinical Pharmacology RAG Chatbot</div>',
                unsafe_allow_html=True)

    if st.session_state.rag_load_error:
        st.error(f"❌ {st.session_state.rag_load_error}")

    elif st.session_state.rag_index is not None:
        _stats = st.session_state.rag_pdf_stats
        st.markdown("""
        <div class="rag-box">
        📖 <b>Clinical Pharmacology</b> reference book is loaded and ready.
        Answers are grounded strictly in the textbook — no hallucination.
        </div>
        """, unsafe_allow_html=True)

        _ca, _cb, _cc = st.columns(3)
        _ca.metric("📚 Reference", "Clinical Pharmacology")
        _cb.metric("📃 Pages",  _stats.get("pages",  "—"))
        _cc.metric("🔍 Chunks", _stats.get("chunks", "—"))

        st.divider()

        st.markdown("**💡 Suggested questions:**")
        _rag_suggestions = [
            "Explain the detailed process of pharmacokinetics (ADME)",
            "How does the cytochrome P450 system affect drug metabolism?",
            "What is the mechanism of action of Beta-blockers?",
            "Explain how NSAIDs inhibit cyclooxygenase and their side effects",
            "What factors determine a drug's ability to cross the blood-brain barrier?",
            "What are the mechanisms behind dangerous drug-drug interactions?",
        ]
        _rag_cols = st.columns(3)
        for i, sug in enumerate(_rag_suggestions):
            if _rag_cols[i % 3].button(sug, key=f"rag_sug_{i}",
                                        use_container_width=True):
                st.session_state.rag_messages.append(
                    {"role": "user", "content": sug}
                )
                with st.spinner("🔍 Searching textbook..."):
                    _resp = get_rag_response(
                        user_message=sug,
                        chat_history=st.session_state.rag_messages[:-1],
                        faiss_index=st.session_state.rag_index,
                        chunks=st.session_state.rag_chunks,
                        groq_api_key=GROQ_API_KEY,
                    )
                st.session_state.rag_messages.append(
                    {"role": "assistant", "content": _resp}
                )
                st.rerun()

        for _msg in st.session_state.rag_messages:
            with st.chat_message(
                _msg["role"],
                avatar="🧑" if _msg["role"] == "user" else "📚",
            ):
                st.markdown(_msg["content"])

        _rag_input = st.chat_input(
            "Ask anything from the Clinical Pharmacology textbook...",
            key="rag_chat_input",
        )
        if _rag_input:
            st.session_state.rag_messages.append(
                {"role": "user", "content": _rag_input}
            )
            with st.chat_message("user", avatar="🧑"):
                st.markdown(_rag_input)

            with st.chat_message("assistant", avatar="📚"):
                with st.spinner("🔍 Searching textbook and generating answer..."):
                    _rag_response = get_rag_response(
                        user_message=_rag_input,
                        chat_history=st.session_state.rag_messages[:-1],
                        faiss_index=st.session_state.rag_index,
                        chunks=st.session_state.rag_chunks,
                        groq_api_key=GROQ_API_KEY,
                    )
                st.markdown(_rag_response)

            st.session_state.rag_messages.append(
                {"role": "assistant", "content": _rag_response}
            )

        if st.session_state.rag_messages:
            if st.button("🗑️ Clear Chat", key="clear_rag_chat"):
                st.session_state.rag_messages = []
                st.rerun()

    else:
        st.info("⏳ Initializing reference book, please wait a moment and refresh...")


st.divider()
st.markdown(
    "<center><small>"
    "AI Pharmacy Assistant &nbsp;·&nbsp; AI Powered &nbsp;·&nbsp; "
    "For educational use only"
    "</small></center>",
    unsafe_allow_html=True,
)
