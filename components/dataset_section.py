import os
import pandas as pd
import plotly.express as px
import streamlit as st
from services.rag import (
    build_or_update_vectorstore_from_dataframe,
    build_or_update_vectorstore_from_pdf,
)

# --- Utility ---
def _load_csv(file):
    df = pd.read_csv(file)
    for cand in ["date", "timestamp", "time", "Date", "Timestamp", "encounter_date"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce")
    return df


def render_dataset_section():
    st.subheader("📊 Data Upload, Exploration & Visualization")

    # --- 1️⃣ Choose from local /data folder ---
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    st.markdown("### 🔽 Choose an existing dataset")
    choice = st.selectbox("Select dataset", ["(none)"] + csv_files, index=0)

    # --- 2️⃣ Or upload your own CSV / PDF ---
    st.markdown("### 📤 Or upload your own file")
    up = st.file_uploader("Upload CSV or PDF", type=["csv", "pdf"], accept_multiple_files=False)

    df = None
    if choice != "(none)":
        path = os.path.join(data_dir, choice)
        df = _load_csv(path)
        st.success(f"✅ Loaded `{choice}` ({len(df)} rows)")
    elif up:
        if up.name.lower().endswith(".csv"):
            df = _load_csv(up)
            st.success(f"✅ Uploaded `{up.name}` ({len(df)} rows)")
        elif up.name.lower().endswith(".pdf"):
            build_or_update_vectorstore_from_pdf(up)
            st.info("Indexed PDF for semantic chat (no visualization available).")
            return  # Skip chart for PDF uploads

    if df is None:
        st.info("Select or upload a dataset to begin.")
        return

    # Save to session + vector store
    st.session_state.df = df
    build_or_update_vectorstore_from_dataframe(df)
    st.session_state.vector_ready = True

    # --- 3️⃣ Data Preview ---
    st.markdown("---")
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(100), use_container_width=True)

    # --- 4️⃣ Quick Visualization on same page ---
    st.markdown("### 📈 Quick Visualization")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].nunique() < len(df) * 0.2 and df[c].nunique() < 50]
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    if not numeric_cols:
        st.warning("No numeric columns found for visualization.")
        return

    viz_type = st.radio(
        "Choose visualization type:",
        ["Trend Line", "Bar Chart", "Pie Chart", "Histogram"],
        horizontal=True,
    )

    if viz_type == "Trend Line":
        if date_cols:
            x_col = st.selectbox("Select date/time column", date_cols)
            y_col = st.selectbox("Select numeric metric", numeric_cols)
            fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, title=f"{y_col} over time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No date/time column found — switch to another chart type.")

    elif viz_type == "Bar Chart":
        y_col = st.selectbox("Select numeric metric", numeric_cols)
        x_col = st.selectbox("Select category (x-axis)", cat_cols or numeric_cols)
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Pie Chart":
        if cat_cols:
            cat_col = st.selectbox("Select category", cat_cols)
            y_col = st.selectbox("Select numeric metric (optional)", ["(count)"] + numeric_cols)
            if y_col == "(count)":
                fig = px.pie(df, names=cat_col, title=f"Distribution of {cat_col}")
            else:
                fig = px.pie(df, names=cat_col, values=y_col, title=f"{y_col} share by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns available for pie chart.")

    elif viz_type == "Histogram":
        y_col = st.selectbox("Select numeric column", numeric_cols)
        fig = px.histogram(df, x=y_col, nbins=20, title=f"Distribution of {y_col}")
        st.plotly_chart(fig, use_container_width=True)
