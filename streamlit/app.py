import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st

# -----------------------------------------------------------------------------
# MUST be first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered",
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SCHEMA_PATH = Path("/app/data/data_schema.json")

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# -----------------------------------------------------------------------------
# Load schema
# -----------------------------------------------------------------------------
@st.cache_resource
def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


schema = load_schema(SCHEMA_PATH)
numerical_features = schema["numerical"]
categorical_features = schema["categorical"]

# -----------------------------------------------------------------------------
# UI Header
# -----------------------------------------------------------------------------
st.title("🎓 Student Performance Prediction")
st.write(
    f"""
This app predicts whether a student will **PASS or FAIL**  
using a machine-learning model served by FastAPI.

**API Endpoint:** `{API_BASE_URL}`
"""
)

st.header("Enter Student Details")

user_input: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Numerical Features
# -----------------------------------------------------------------------------
st.subheader("Numerical Features")

for feature, stats in numerical_features.items():
    min_val = float(stats["min"])
    max_val = float(stats["max"])
    median_val = float(stats["median"])

    label = feature.replace("_", " ").title()
    help_text = (
        f"Min: {min_val:.2f}, Max: {max_val:.2f}, Median: {median_val:.2f}"
    )

    step = 0.1
    if max_val - min_val < 20:
        step = 0.1
    if max_val - min_val < 10:
        step = 0.01

    user_input[feature] = st.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=median_val,
        step=step,
        help=help_text,
        key=feature,
    )

# -----------------------------------------------------------------------------
# Categorical Features
# -----------------------------------------------------------------------------
st.subheader("Categorical Features")

for feature, info in categorical_features.items():
    unique_values = info["unique_values"]
    value_counts = info["value_counts"]

    default_value = max(value_counts, key=value_counts.get)
    default_index = unique_values.index(default_value)

    label = feature.replace("_", " ").title()

    user_input[feature] = st.selectbox(
        label,
        options=unique_values,
        index=default_index,
        key=feature,
        help=f"Distribution: {value_counts}",
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
if st.button("🔮 Predict Result", type="primary"):
    payload = {"instances": [user_input]}

    with st.spinner("Calling prediction API..."):
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

                with st.expander("📋 Input Summary"):
                    st.json(user_input)

st.markdown("---")
st.caption(
    f"""
📁 Schema: `{SCHEMA_PATH}`  
🌐 API: `{API_BASE_URL}`
"""
)
