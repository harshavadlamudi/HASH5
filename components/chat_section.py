import streamlit as st
from services.bedrock_client import llm_answer
from services.rag import (
    semantic_search_context,
    build_or_update_vectorstore_from_pdf,
    build_or_update_vectorstore_from_dataframe,
)
import pandas as pd
from PIL import Image
import io


SYSTEM_PATIENT = (
    "You are Healogue, a friendly health assistant for patients. "
    "Explain medical terms simply, avoid jargon, and be reassuring."
)
SYSTEM_CLINICIAN = (
    "You are Healogue, a concise assistant for clinicians. "
    "Use precise language, cite values and time frames, and avoid speculation."
)


def _system_prompt():
    return SYSTEM_CLINICIAN if st.session_state.mode == "Clinician" else SYSTEM_PATIENT


def _prepare_context(user_text: str):
    ctx_docs = semantic_search_context(user_text)
    return "\n\n".join(ctx_docs) if ctx_docs else ""

def render_chat_section():
    st.subheader("💬 Chat with Healogue")

    # --- Initialize session state safely ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "greeted" not in st.session_state:
        st.session_state.greeted = False

    # --- Add greeting only once ---
    if not st.session_state.greeted:
        if st.session_state.mode == "Clinician":
            greeting = (
                "Hello Doctor! I'm Healogue, your concise health data assistant. "
                "How can I support your patient analysis today?"
            )
        else:
            greeting = (
                "Hello! I'm Healogue, your friendly health assistant. "
                "I'm here to explain medical terms simply and help you understand your reports."
            )
        st.session_state.chat_history.append(("assistant", greeting))
        st.session_state.greeted = True

    # --- File Upload (always visible) ---
    st.markdown("### 📂 Drop a file to analyze (PDF, CSV, or Image)")
    uploaded_file = st.file_uploader(
        "Upload a healthcare file (PDF, CSV, PNG, JPG)",
        type=["pdf", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    # --- Process new file uploads ---
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        st.session_state.vector_ready = False

        if filename.endswith(".pdf"):
            build_or_update_vectorstore_from_pdf(uploaded_file)
            st.success(f"✅ Indexed `{uploaded_file.name}` for question answering.")
            st.session_state.vector_ready = True

        elif filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            build_or_update_vectorstore_from_dataframe(df)
            st.success(f"✅ Loaded `{uploaded_file.name}` with {len(df)} rows.")
            st.session_state.vector_ready = True

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            try:
                import pytesseract
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=600)
                text = pytesseract.image_to_string(image)
                if text.strip():
                    st.info("🧠 Extracted text from image, adding to context.")
                    from services.rag import _collection, _embed
                    embeddings = _embed([text])
                    _collection.upsert(
                        documents=[text],
                        embeddings=embeddings,
                        ids=[f"image-{uploaded_file.name}"],
                    )
                    st.session_state.vector_ready = True
                else:
                    st.warning("No readable text found in the image.")
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.info("Tip: Install Tesseract OCR or enable Textract on AWS for better results.")

        st.markdown("---")

    # --- Display chat history ---
    chat = st.container()
    for role, msg in st.session_state.chat_history:
        klass = "hlg-user" if role == "user" else "hlg-bot"
        chat.markdown(f"<div class='{klass}'>{msg}</div>", unsafe_allow_html=True)

    # --- Chat input box ---
    with st.form(key="chat_form", clear_on_submit=True):
        user_text = st.text_input(
            "Ask a question about your data, report, or image...",
            key="chat_input",
        )
        submitted = st.form_submit_button("Send", type="primary")

    # --- Process chat query ---
    if submitted and user_text:
        st.session_state.chat_history.append(("user", user_text))
        with st.spinner("Analyzing..."):
            system_prompt = _system_prompt()
            context = _prepare_context(user_text)
            df_json = None

            if st.session_state.get("df") is not None:
                try:
                    df_json = st.session_state.df.to_json(orient="records")[:20000]
                except Exception:
                    df_json = None

            final_prompt = {
                "system": system_prompt,
                "question": user_text,
                "mode": st.session_state.mode,
                "context": context,
                "data_preview": df_json,
                "instruction": (
                    "If the user asks about the uploaded file or image, summarize its content. "
                    "If the file is a CSV, reference its metrics. If it’s an image, describe "
                    "any detected text or visible data. Use the context when relevant."
                ),
            }

            answer = llm_answer(final_prompt)
            st.session_state.chat_history.append(("assistant", answer))
            st.rerun()

    # --- Hint message ---
    st.info("Try: 'Summarize my uploaded report' or 'What does the image text say?'")
