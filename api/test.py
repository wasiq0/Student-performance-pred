import housing_pipeline
import joblib

print("Loading model...")
model = joblib.load("models/global_best_model_optuna.pkl")
print("Model loaded:", type(model))
