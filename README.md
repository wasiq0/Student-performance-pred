# 🎓 Student Performance Prediction – End-to-End Machine Learning System

## Live deployed apps:
🔵 Streamlit Frontend: https://student-performance-pred-1.onrender.com/
🟢 FastAPI Backend: https://student-performance-pred-szzj.onrender.com/
This project predicts whether a student will Pass or Fail based on behavioral, academic, and lifestyle factors.
It includes a full ML pipeline, FastAPI backend, Streamlit frontend, Docker orchestration, and full cloud deployment on Render.
## 🚀 Features
### ✅ Machine Learning Pipeline
SQLite database creation from CSV
Data cleaning, encoding, and feature engineering
PCA vs non-PCA experiment variants
16 experiments (LogReg, Ridge, HistGradientBoosting, XGBoost)
Optuna hyperparameter tuning
MLflow experiment tracking
Best model saved in /api/models/

### ✅ FastAPI Backend
POST /predict endpoint
GET /health endpoint
Loads best Optuna-tuned model
Fully containerized via Docker

### ✅ Streamlit Frontend
User input form
Sends request to FastAPI
Displays prediction results
Dockerized + deployed on Render

### ✅ Cloud Deployment (Render)
FastAPI Docker service
Streamlit Docker service
Environment variable linking frontend ↔ backend
Publicly accessible URLs


### 👨‍🎓 Author

Wasiq Nabi Bakhsh
MS Engineering Data Science
University at Buffalo

🔗 GitHub: https://github.com/wasiq0
