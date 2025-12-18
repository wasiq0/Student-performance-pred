# api/app.py
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import housing_pipeline  # CRITICAL: must be imported BEFORE loading model

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path("/app/models/best_optuna_classifier.joblib")

app = FastAPI(title="Student Performance API")

print(f"Loading model from {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[int]
    count: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Convert incoming JSON to DataFrame
    X = pd.DataFrame(req.instances)

    # 🔧 QUICK PATCH — model pipeline expects this column
    if "student_id" not in X.columns:
        X["student_id"] = 0

    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        predictions=[int(p) for p in preds],
        count=len(preds)
    )
