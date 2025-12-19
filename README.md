🎓 Student Performance Prediction — End-to-End ML System
FastAPI API • Streamlit UI • Docker • Optuna • MLflow • Render Deployment

🔗 Live Streamlit App:
➡️ https://student-performance-pred-1.onrender.com/

🔗 Live FastAPI Endpoint:
➡️ https://student-performance-pred-szzj.onrender.com/

This project is an end-to-end machine learning system that predicts whether a student will Pass or Fail based on behavioral, academic, and lifestyle factors.
The entire workflow — data → ML → API → UI → cloud deployment — is fully automated and containerized.

📌 Features
✅ Complete ML Pipeline
SQLite database from CSV
Data cleaning + preprocessing
PCA and non-PCA workflows
16 experiments (LogReg, XGBoost, Ridge, HistGB, PCA versions)
Optuna hyperparameter tuning
MLflow experiment tracking
Best model saved to /api/models/
✅ Backend (FastAPI)
Predict endpoint (POST /predict)
Health check (GET /health)
Loads Optuna-tuned best classifier
Fully containerized with Docker
✅ Frontend (Streamlit)
Clean and interactive UI
Sends inputs to the FastAPI service
Displays results, prediction summary
Runs on Render using Docker
✅ Cloud Deployment (Render)
1 Docker service for API
1 Docker service for Streamlit
Environment variable for API URL
Both apps running publicly on free tier
🏗️ Project Structure
student-performance-pred/
├── api/
│   ├── app.py
│   ├── Dockerfile
│   ├── housing_pipeline.py
│   ├── requirements.txt
│   └── models/best_optuna_classifier.joblib
│
├── streamlit/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── data/
│   ├── student_performance.csv
│   ├── student_performance.db
│   └── data_schema.json
│
├── notebooks/
│   ├── 01_create_database.ipynb
│   ├── 02_train_model_without_optuna.ipynb
│   ├── 03_train_model_with_optuna.ipynb
│   └── 04_generate_streamlit_options.ipynb
│
├── models/
├── reports/
├── mlruns/
│
├── docker-compose.yml
├── test_inference.py
├── housing_pipeline.py
└── README.md

📊 Dataset Overview

The dataset contains 20,000 rows and includes:
Feature	Description
age	Student age
gender	Gender identity
course	Program enrolled
study_hours	Daily hours spent studying
class_attendance	Attendance %
internet_access	yes/no
sleep_hours	Hours of sleep
sleep_quality	Poor/Average/Good
study_method	Self-study / Group / Coaching
facility_rating	low/medium/high
exam_difficulty	easy/moderate/hard
exam_score	Actual numeric score
target	Pass (1) / Fail (0)

🚀 Running Locally (Optional)
1. Build images
docker-compose build
2. Run services
docker-compose up
3. Visit apps
Streamlit: http://localhost:8501
API: http://localhost:8000

⚙️ API Documentation
🔹 Health Check
GET /health
Example:
{"status": "ok"}
🔹 Predict
POST /predict
Example Request:
{
  "instances": [
    {
      "age": 20,
      "gender": "male",
      "study_hours": 3,
      "class_attendance": 85,
      "sleep_hours": 7,
      "sleep_quality": "good",
      "internet_access": "yes",
      "course": "b.tech",
      "exam_difficulty": "moderate",
      "study_method": "self-study",
      "facility_rating": "high",
      "exam_score": 60
    }
  ]
}

Example Response:
{
  "predictions": [1],
  "count": 1
}

🌐 Cloud Deployment (Render)
✔ Backend (FastAPI)
Docker environment
Root: /api
Exposes port 8000
Public URL: https://student-performance-pred-szzj.onrender.com/
✔ Frontend (Streamlit)
Docker environment
Root: /streamlit
Environment variable: API_URL=https://student-performance-pred-szzj.onrender.com


Public URL:
https://student-performance-pred-1.onrender.com/

🧪 Model Testing
Run: python api/test.py
Or use curl:
curl -X POST "https://student-performance-pred-szzj.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"instances":[{...}]}'


👨‍💻 Author
Wasiq Nabi Bakhsh
MS Engineering Data Science
University at Buffalo

GitHub: https://github.com/wasiq0

Project: Student Performance Prediction
