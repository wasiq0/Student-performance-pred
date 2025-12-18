FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY housing_pipeline.py .
COPY models/ /app/models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
