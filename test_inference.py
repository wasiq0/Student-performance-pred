# test_model_inference.py
"""
Test script to verify the saved STUDENT CLASSIFICATION model
can be loaded and used for inference.
"""

import joblib
import pandas as pd
from pathlib import Path


def test_model(model_path):
    """Load model and test classification inference."""
    print(f"\n{'='*70}")
    print(f"Testing model: {model_path}")
    print("=" * 70)

    try:
        # Load model
        print("Loading model...", end=" ")
        model = joblib.load(model_path)
        print("✓")

        if hasattr(model, "named_steps"):
            print(f"Pipeline steps: {list(model.named_steps.keys())}")

        # Create sample STUDENT data (matches DB + Streamlit)
        sample_data = pd.DataFrame([
            {
                "age": 20,
                "gender": "male",
                "internet_access": "yes",
                "sleep_hours": 7.5,
                "sleep_quality": "good",
                "class_attendance": 85.0,
                "course": "b.tech",
                "study_method": "self-study",
                "facility_rating": "high",
                "exam_difficulty": "moderate",
                "study_hours": 4.5,
            },
            {
                "age": 22,
                "gender": "female",
                "internet_access": "no",
                "sleep_hours": 5.0,
                "sleep_quality": "poor",
                "class_attendance": 55.0,
                "course": "b.sc",
                "study_method": "online videos",
                "facility_rating": "low",
                "exam_difficulty": "hard",
                "study_hours": 2.0,
            },
        ])

        # Run inference
        print("Running inference...", end=" ")
        preds = model.predict(sample_data)
        print("✓")

        # Display results
        print("\nPredictions:")
        for i, p in enumerate(preds, 1):
            label = "PASS" if p == 1 else "FAIL"
            print(f"  Sample {i}: {label} ({p})")

        print(f"\n✓ {Path(model_path).name} — SUCCESS")
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


def main():
    """Test the saved classification model."""
    model_path = "models/global_best_model_optuna.pkl"

    print("Testing student classification model...")
    success = test_model(model_path)

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("PASS" if success else "FAIL")
    print("=" * 70)

    return success


if __name__ == "__main__":
    exit(0 if main() else 1)
