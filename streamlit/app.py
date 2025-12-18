import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

# =============================================================================
# Streamlit page config (MUST be first)
# =============================================================================
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered",
)

# =============================================================================
# Configuration
# =============================================================================

# Schema file lives INSIDE the streamlit folder
SCHEMA_PATH = Path(__file__).parent / "data_schema.json"

# 🚨 IMPORTANT: NO localhost fallback in cloud
# API_URL MUST be provided via Render Environment Variables
API_BASE_URL = os.environ["API_URL"]
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# Debug (helps during grading/demo)
st.write("🔗 Using API:", API_BASE_URL)

# =============================================================================
# Load schema
# =============================================================================
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


schema = load_schema(SCHEMA_PATH)
numerical_features = schema["numerical"]
categorical_features = schema["categorical"]

# =============================================================================
# UI Header
# =============================================================================
st.title("🎓 Student Performance Prediction")

st.markdown(
    """
This application predicts whether a student will **PASS or FAIL**
using a **machine-learning classification model** served by **FastAPI**.

The frontend (Streamlit) communicates with the backend (FastAPI)
over HTTP using Docker-based deployment.
"""
)

st.markdown(f"**FastAPI Endpoint:** `{API_BASE_URL}`")

st.divider()
st.header("📝 Enter Student Details")

user_input: Dict[str, Any] = {}

# =============================================================================
# Numerical Features
# =============================================================================
st.subheader("🔢 Numerical Features")

for feature, stats in numerical_features.items():
    min_val = float(stats["min"])
    max_val = float(stats["max"])
    median_val = float(stats["median"])

    label = feature.replace("_", " ").title()
    help_text = f"Min: {min_val}, Max: {max_val}, Median: {median_val}"

    step = 0.01 if max_val - min_val < 10 else 0.1

    user_input[feature] = st.number_input(
        label=label,
        min_value=min_val,
        max_value=max_val,
        value=median_val,
        step=step,
        help=help_text,
        key=feature,
    )

# =============================================================================
# Categorical Features
# =============================================================================
st.subheader("🧩 Categorical Features")

for feature, info in categorical_features.items():
    options = info["unique_values"]
    counts = info["value_counts"]

    default_value = max(counts, key=counts.get)
    default_index = options.index(default_value)

    label = feature.replace("_", " ").title()

    user_input[feature] = st.selectbox(
        label=label,
        options=options,
        index=default_index,
        help=f"Value distribution: {counts}",
        key=feature,
    )

# =============================================================================
# Prediction
# =============================================================================
st.divider()

if st.button("🔮 Predict Result", type="primary"):
    payload = {"instances": [user_input]}

    with st.spinner("Calling FastAPI prediction service..."):
        try:
            response = requests.post(
                PREDICT_ENDPOINT,
                json=payload,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API request failed: {e}")
        else:
            if response.status_code != 200:
                st.error(
                    f"❌ API error ({response.status_code}): {response.text}"
                )
            else:
                result = response.json()
                prediction = result["predictions"][0]

                st.success("✅ Prediction Successful")

                st.subheader("📊 Prediction Result")

                if prediction == 1:
                    st.metric("Predicted Outcome", "PASS ✅")
                else:
                    st.metric("Predicted Outcome", "FAIL ❌")

                with st.expander("📋 Submitted Input"):
                    st.json(user_input)

# =============================================================================
# Footer
# =============================================================================
st.divider()
st.caption(
    f"""
📁 Schema File: `{SCHEMA_PATH.name}`  
🌐 FastAPI Service: `{API_BASE_URL}`
"""
)
