import streamlit as st


def render_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("### 🩺 Healogue")
    with col2:
        st.markdown("#### Chat with your health data – safely and simply")


    with st.sidebar:
        st.markdown("### Mode")
    st.session_state.mode = st.radio("User type", ["Patient", "Clinician"], index=0, horizontal=True)
    st.markdown("---")