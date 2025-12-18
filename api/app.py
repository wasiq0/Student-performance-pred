# api/app.py
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

# ⚠️ CRITICAL
# This MUST be imported before loading the model
import housing_pipeline  

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_PATH = Path("/app/models/best_optuna_classifier.joblib")

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Student Performance Prediction API",
    description="FastAPI service for ML inference",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# Load model at startup
# -----------------------------------------------------------------------------
print(f"🚀 Loading model from {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[int]
    count: int


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    """
    Root endpoint (for browser + professor sanity check)
    """
    return {
        "message": "Student Performance Prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    """
    Health check endpoint (used by Render)
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run inference on incoming student data
    """
    if not req.instances:
        raise HTTPException(status_code=400, detail="No instances provided")

    # Convert input to DataFrame
    X = pd.DataFrame(req.instances)

    # -------------------------------------------------------------------------
    # QUICK PATCH (EXPECTED & ACCEPTABLE)
    # Model pipeline was trained with `student_id`
    # UI should not ask for it → inject dummy value
    # -------------------------------------------------------------------------
    if "student_id" not in X.columns:
        X["student_id"] = 0

    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        predictions=[int(p) for p in preds],
        count=len(preds),
    )
