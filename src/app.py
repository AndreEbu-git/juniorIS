import streamlit as st
import pandas as pd
from io import BytesIO

# Title
st.set_page_config(
    page_title="Business Performance Forecaster",
    page_icon="📊",
    layout="wide"
)
st.title("Dynamic Business Performance Forecaster")
st.markdown("Upload your business data to start forecasting and scenario planning")

# Tabs
tab1, tab2 = st.tabs(["📁 Upload Dataset", "📋 Data Summary"])

# --- Feature 1: Load and Preprocess ---
with tab1:
    st.header("Load Business Dataset")
    st.markdown("Upload your business data (CSV or Excel format) ")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Supported formats: CSV, Excel(.xlsx, .xls)"
    )

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            else:
                df = pd.read_excel(uploaded_file)

            # Start Basic preprocessing
            df = df.dropna(thresh=len(df.columns)*0.7) # Drops rows with a lot of missing data
            if 'Date' in df.columns or 'date' in df.columns.lower():
                date_col = next(c for c in df.columns if 'date' in c.lower())
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

            st.session_state['df'] = df
            st.success(f"Data loaded! Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            