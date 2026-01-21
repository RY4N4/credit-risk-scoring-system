# 🏦 Credit Risk Scoring System

> **Production-Ready ML System with Interactive Web UI**  
> Real-time loan default prediction using XGBoost, FastAPI, and Streamlit

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-green)
![Streamlit](https://img.shields.io/badge/Streamlit-red)
![XGBoost](https://img.shields.io/badge/XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Business Problem

Loan defaults are a major source of loss for lending institutions.  
Traditional underwriting processes are:

- ⏱️ Slow and manual  
- 💸 Costly when defaults occur  
- ⚖️ Inconsistent across decisions  
- 📉 Hard to scale  

### Objective
Build a **real-time, automated credit risk scoring system** that predicts loan default risk and supports business-aligned lending decisions.

---

## 🚀 Solution Overview

This project implements an **end-to-end machine learning system** that:

1. Ingests loan application data via API or web UI  
2. Applies consistent preprocessing and feature engineering  
3. Predicts default probability using an XGBoost model  
4. Returns actionable decisions:
   - **APPROVE**
   - **REVIEW**
   - **REJECT**

The system is deployed as a **FastAPI service** with an **interactive Streamlit UI**, delivering predictions in **under 100ms**.

---

## 🎨 Interactive Web UI

A Streamlit-based frontend for real-time credit risk assessment.

### Features
- Interactive input forms with validation  
- Live predictions via FastAPI backend  
- Visual risk categorization (LOW / MEDIUM / HIGH)  
- Business-readable explanations  
- Responsive design for desktop and mobile  

### Run the UI
```bash
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
Techniques Used
Feature engineering (debt-to-income, loan-to-income, employment stability)
Class imbalance handling using:
SMOTE oversampling
scale_pos_weight
Cost-sensitive threshold optimization
📊 Model Performance
Metric	Logistic Regression	XGBoost
ROC-AUC	0.76	0.82
Precision (Default)	0.68	0.74
Recall (Default)	0.42	0.51
F1-score	0.52	0.60
Threshold Optimization
The decision threshold was optimized based on business cost, where false negatives (approving bad loans) are significantly more expensive than false positives.
🌐 API (FastAPI)
Endpoints
POST /predict-risk – Predict loan default risk
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
FastAPI with Pydantic validation
Consistent preprocessing for training and inference
Model artifact serialization
Structured logging and health checks
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
│   └── evaluation plots
└── README.md
📚 Dataset
Source: Lending Club (Kaggle)
Size: ~2.2M loan records
Target: Loan default (binary)
Class Imbalance: ~10–20% defaults
🎓 Learning Outcomes
This project demonstrates:
End-to-end ML system development
Handling imbalanced classification problems
Feature engineering for credit risk
Cost-sensitive decision making
Production ML deployment using FastAPI
API-based model serving with validation
