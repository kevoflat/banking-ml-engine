# Banking Risk ML Engine

Python FastAPI ML engine for credit risk scoring and fraud detection.
Serves as the ML backend for the Banking Risk Assessment System.

## Endpoints
- `GET /health` — health check
- `POST /score/loan` — credit risk scoring
- `POST /score/transaction` — fraud detection scoring

## Stack
Python · FastAPI · scikit-learn · numpy · pandas · uvicorn

## Run
pip install fastapi uvicorn scikit-learn numpy pandas joblib
python -m uvicorn main:app --reload --port 8000