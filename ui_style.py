def inject_healogue_style():
    import streamlit as st
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px;}
        .stCard, .stMarkdown, .stDataFrame {border-radius: 16px;}
        .hlg-user{background:#e2f2ff;padding:12px 14px;border-radius:14px;margin:6px 0;}
        .hlg-bot{background:#f1f5f9;padding:12px 14px;border-radius:14px;margin:6px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )
