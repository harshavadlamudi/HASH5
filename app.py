import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from components.header import render_header
from components.dataset_section import render_dataset_section
from components.chat_section import render_chat_section
from ui_style import inject_healogue_style
from theme_plotly import apply_healogue_plotly_theme

# Streamlit config
st.set_page_config(page_title="Healogue - AI Health Chat", layout="wide")
apply_healogue_plotly_theme()
inject_healogue_style()

# Session defaults
if "mode" not in st.session_state:
    st.session_state.mode = "Patient"
if "df" not in st.session_state:
    st.session_state.df = None
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Header
render_header()

# Sidebar navigation
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Dataset", "Chat"], index=0)

# Pages
if page == "Dataset":
    render_dataset_section()
else:
    render_chat_section()

st.markdown(
    "<div style='text-align:center;color:#8a8a8a;margin-top:2rem'>"
    "Built for AWS Bayern Health Hackathon"
    "</div>",
    unsafe_allow_html=True,
)
