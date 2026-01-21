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
