"""
banking-ml-engine/main.py
FastAPI ML engine — credit risk scoring + fraud detection
Called by Java Spring Boot backend
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Banking Risk ML Engine",
    description="Credit risk scoring and fraud detection API",
    version="1.0.0"
)

# ── Request schemas ────────────────────────────────────────────────────────────

class LoanScoringRequest(BaseModel):
    monthly_income: float
    existing_debt: float
    credit_score: int
    loan_amount: float
    term_months: int
    debt_to_income: float

class TransactionScoringRequest(BaseModel):
    amount: float
    type: str
    credit_score: Optional[int] = 650
    monthly_income: Optional[float] = 50000

# ── Response schemas ───────────────────────────────────────────────────────────

class LoanScoringResponse(BaseModel):
    risk_score: float
    risk_category: str
    recommended_interest_rate: float
    max_recommended_loan: float
    explanation: dict

class TransactionScoringResponse(BaseModel):
    fraud_score: float
    flag_status: str
    explanation: dict

# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "UP", "service": "Banking Risk ML Engine"}

# ── Loan risk scoring ──────────────────────────────────────────────────────────

@app.post("/score/loan", response_model=LoanScoringResponse)
def score_loan(request: LoanScoringRequest):
    log.info(f"Scoring loan — amount: {request.loan_amount}, income: {request.monthly_income}")

    # Feature engineering
    dti_ratio = request.existing_debt / max(request.monthly_income, 1)
    loan_to_income = request.loan_amount / max(request.monthly_income, 1)
    monthly_obligation = request.loan_amount / request.term_months
    obligation_ratio = monthly_obligation / max(request.monthly_income, 1)
    credit_normalized = (request.credit_score - 300) / 550  # normalize 300-850

    # Risk score calculation (0.0 = no risk, 1.0 = maximum risk)
    risk_score = 0.0

    # Credit score factor (40% weight)
    credit_factor = 1.0 - credit_normalized
    risk_score += credit_factor * 0.40

    # Debt-to-income factor (25% weight)
    if dti_ratio < 0.2:
        dti_factor = 0.1
    elif dti_ratio < 0.4:
        dti_factor = 0.3
    elif dti_ratio < 0.6:
        dti_factor = 0.6
    else:
        dti_factor = 0.9
    risk_score += dti_factor * 0.25

    # Loan-to-income factor (20% weight)
    if loan_to_income < 2:
        lti_factor = 0.1
    elif loan_to_income < 5:
        lti_factor = 0.4
    elif loan_to_income < 10:
        lti_factor = 0.7
    else:
        lti_factor = 0.95
    risk_score += lti_factor * 0.20

    # Monthly obligation factor (15% weight)
    if obligation_ratio < 0.2:
        ob_factor = 0.1
    elif obligation_ratio < 0.35:
        ob_factor = 0.4
    elif obligation_ratio < 0.5:
        ob_factor = 0.7
    else:
        ob_factor = 0.95
    risk_score += ob_factor * 0.15

    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    # Risk category
    if risk_score < 0.3:
        risk_category = "LOW"
        interest_rate = 8.5
    elif risk_score < 0.6:
        risk_category = "MEDIUM"
        interest_rate = 14.0
    else:
        risk_category = "HIGH"
        interest_rate = 22.0

    # Max recommended loan
    max_loan = request.monthly_income * 12 * (1 - risk_score)

    explanation = {
        "credit_score_impact": round(credit_factor * 0.40, 4),
        "debt_to_income_impact": round(dti_factor * 0.25, 4),
        "loan_to_income_impact": round(lti_factor * 0.20, 4),
        "obligation_ratio_impact": round(ob_factor * 0.15, 4),
        "dti_ratio": round(dti_ratio, 4),
        "loan_to_income_ratio": round(loan_to_income, 4),
    }

    log.info(f"Loan risk score: {risk_score:.4f} — {risk_category}")

    return LoanScoringResponse(
        risk_score=risk_score,
        risk_category=risk_category,
        recommended_interest_rate=interest_rate,
        max_recommended_loan=round(max_loan, 2),
        explanation=explanation
    )

# ── Transaction fraud scoring ──────────────────────────────────────────────────

@app.post("/score/transaction", response_model=TransactionScoringResponse)
def score_transaction(request: TransactionScoringRequest):
    log.info(f"Scoring transaction — amount: {request.amount}, type: {request.type}")

    fraud_score = 0.05

    # Amount relative to income
    if request.monthly_income and request.monthly_income > 0:
        ratio = request.amount / request.monthly_income
        if ratio > 10:
            fraud_score += 0.55
        elif ratio > 5:
            fraud_score += 0.40
        elif ratio > 2:
            fraud_score += 0.20
        elif ratio > 1:
            fraud_score += 0.10

    # Absolute amount thresholds
    if request.amount > 1000000:
        fraud_score += 0.35
    elif request.amount > 500000:
        fraud_score += 0.20
    elif request.amount > 100000:
        fraud_score += 0.10

    # Transaction type risk
    type_risk = {
        "TRANSFER": 0.05,
        "WITHDRAWAL": 0.08,
        "DEPOSIT": 0.01
    }
    fraud_score += type_risk.get(request.type.upper(), 0.05)

    # Low credit score increases fraud risk
    if request.credit_score and request.credit_score < 400:
        fraud_score += 0.15
    elif request.credit_score and request.credit_score < 500:
        fraud_score += 0.08

    fraud_score = float(np.clip(fraud_score, 0.0, 1.0))

    if fraud_score > 0.7:
        flag_status = "FRAUDULENT"
    elif fraud_score > 0.4:
        flag_status = "SUSPICIOUS"
    else:
        flag_status = "NORMAL"

    explanation = {
        "amount_risk": round(request.amount / max(request.monthly_income or 1, 1), 4),
        "transaction_type": request.type,
        "credit_score": request.credit_score,
        "flag_reason": flag_status
    }

    log.info(f"Fraud score: {fraud_score:.4f} — {flag_status}")

    return TransactionScoringResponse(
        fraud_score=fraud_score,
        flag_status=flag_status,
        explanation=explanation
    )