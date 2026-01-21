🏦 Credit Risk Scoring System
Production-Ready ML System with Interactive Web UI
Real-time loan default prediction using XGBoost, FastAPI, and Streamlit





🎯 Business Problem
Loan defaults are one of the largest cost drivers in consumer lending.
Manual underwriting is:
⏱️ Slow (days to weeks)
💸 Expensive (avg. loss ≈ $9,000 per default)
⚖️ Inconsistent (subjective decisions)
📉 Inefficient at scale
Objective
Build a real-time, automated credit risk scoring system that:
Predicts default probability accurately
Optimizes decisions based on business cost
Is deployable as a production ML service
🚀 Solution Overview
This project implements an end-to-end ML system that:
Ingests loan applications via API or UI
Applies consistent preprocessing & feature engineering
Predicts default risk using XGBoost
Returns a business-aligned decision:
APPROVE
REVIEW
REJECT
All predictions are served in <100ms latency via FastAPI.
🎨 Interactive Web UI
A Streamlit-based frontend for real-time risk assessment.
Features
Interactive input forms with validation
Live predictions via FastAPI backend
Visual risk indicator (LOW / MEDIUM / HIGH)
Business-readable explanations
Responsive design (desktop + mobile)
Run UI
streamlit run app.py
🏗️ System Architecture
Raw Data (Lending Club)
        ↓
Preprocessing Pipeline
        ↓
Feature Engineering
        ↓
Model Training (Logistic + XGBoost)
        ↓
Evaluation & Threshold Optimization
        ↓
FastAPI Inference Service
        ↓
Streamlit Web UI / REST API
🧠 Machine Learning Details
Models
Logistic Regression (baseline)
XGBoost (production model)
Key Techniques
Feature engineering (DTI, loan-to-income, employment stability)
Class imbalance handling:
SMOTE
scale_pos_weight
Cost-sensitive threshold optimization
📊 Model Performance
Metric	Logistic Regression	XGBoost
ROC-AUC	0.76	0.82
Precision (Default)	0.68	0.74
Recall (Default)	0.42	0.51
F1-score	0.52	0.60
Why Threshold ≠ 0.5
False negatives (approving bad loans) are ~5× more costly than false positives.
The decision threshold was optimized to minimize expected financial loss, not accuracy.
🌐 API (FastAPI)
Endpoints
POST /predict-risk – Predict default probability
GET /health – Service health check
Example Request
{
  "loan_amnt": 15000,
  "term": "36 months",
  "int_rate": 12.5,
  "annual_inc": 75000,
  "dti": 18.5,
  "grade": "B",
  "emp_length": "5 years"
}
Example Response
{
  "risk_score": 0.23,
  "decision": "APPROVE",
  "risk_category": "MEDIUM"
}
🔐 Production Readiness
FastAPI + Pydantic validation
Preprocessing consistency (train = inference)
Model artifact versioning
Structured logging
Health checks
Sub-100ms inference latency
📁 Project Structure
credit-risk-scoring/
│
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   ├── main.py
│   └── schema.py
├── models/
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
├── reports/
│   └── metrics & plots
└── README.md
📚 Dataset
Source: Lending Club (Kaggle)
Records: ~2.2M loans
Target: Loan default (binary)
Imbalance: ~10–20% defaults
🎓 Learning Outcomes
This project demonstrates:
End-to-end ML system design
Production ML deployment
Cost-sensitive decision making
Handling imbalanced classification
Feature engineering for financial risk
API-based model serving
🧠 Resume-Ready Summary
Built a production-grade credit risk scoring system using XGBoost and FastAPI, achieving 82% ROC-AUC and serving real-time loan decisions with sub-100ms latency through an interactive Streamlit UI.
⭐ If this helped you
Give the repo a ⭐ — it helps visibility and credibility.
