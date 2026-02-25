import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dataloader import DataLoader, DataPreprocessor, DataValidator


# Title
st.set_page_config(
    page_title="Business Performance Forecaster",
    page_icon="📊",
    layout="wide"
)
if 'data_loader' not in st.session_state:
    st.session_state.dataloader = DataLoader()


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
        
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            else:
                df = pd.read_excel(uploaded_file)

            st.success(f" Successfully loaded {len(df)} rows")

            # Start Basic preprocessing
            st.subheader("Data Validation")
            validator = DataValidator(min_rows=10)
            is_valid, issues = validator.validate_dataset(df)

            if is_valid:
                st.success("Dataset passes basic validation")

            else:
                st.warning("Validation Issues found")
                for issue in issues:
                    st.write(f"- {issue}")

            if validator.validation_results.get('warnings'):
                with st.expander("⚠️ Data Quality Warnings"):
                    for warning in validator.validation_results['warnings']:
                        st.write(f"- {warning}")

            st.session_state['df'] = df
            st.success(f"Data loaded! Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# --- Data Preview ---
with tab2:
    if 'df' in st.session_state:
        st.subheader("Data Preview")
        st.dataframe(st.session_state['df'].head(10))
        
        st.subheader("Quick Stats")
        st.write(st.session_state['df'].describe())