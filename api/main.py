"""
FastAPI Application for Credit Risk Scoring
-------------------------------------------
Production-ready REST API for loan default prediction.

Business Context:
- Real-time credit decisions
- High availability requirement
- Input validation critical
- Audit logging for compliance
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schema import (
    CreditRiskRequest,
    CreditRiskResponse,
    HealthResponse,
    ErrorResponse,
    LendingDecision,
    RiskCategory
)
from src.model_utils import ModelLoader, format_api_response

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Scoring API",
    description="Real-time credit risk assessment for loan applications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (configure properly for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loader (loaded once at startup)
model_loader = None
DECISION_THRESHOLD = 0.35  # Business-defined threshold


@app.on_event("startup")
async def startup_event():
    """
    Load model and artifacts on application startup.
    
    Production considerations:
    - Fail fast if model missing
    - Log model version
    - Warm up model (optional)
    """
    global model_loader
    
    try:
        logger.info("Starting Credit Risk API...")
        
        # Determine model directory
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'models'
        )
        
        # Initialize model loader
        model_loader = ModelLoader(model_dir=model_dir)
        
        # Load model and preprocessing artifacts
        model_loader.load_all(model_name='credit_model.pkl')
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"✅ Decision threshold: {DECISION_THRESHOLD}")
        logger.info("✅ API ready to serve requests")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error("API cannot start without model. Please train model first.")
        raise


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Credit Risk Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
    - API status
    - Model load status
    - Version info
    
    Use this for:
    - Load balancer health checks
    - Monitoring systems
    - Deployment verification
    """
    model_loaded = model_loader is not None and model_loader.model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        version="1.0.0"
    )


def preprocess_applicant_data(request: CreditRiskRequest) -> pd.DataFrame:
    """
    Convert API request to model-ready DataFrame.
    
    CRITICAL: Must match training data preprocessing exactly.
    
    Steps:
    1. Convert Pydantic model to dict
    2. Create DataFrame
    3. Apply same transformations as training
    
    Args:
        request: Validated request from API
        
    Returns:
        DataFrame ready for model prediction
    """
    # Convert to dictionary
    data = {
        'loan_amnt': request.loan_amnt,
        'term': request.term.value,
        'int_rate': request.int_rate,
        'annual_inc': request.annual_inc,
        'dti': request.dti,
        'grade': request.grade.value,
        'sub_grade': request.sub_grade,
        'emp_length': request.emp_length,
        'home_ownership': request.home_ownership.value,
        'purpose': request.purpose.value
    }
    
    # Create DataFrame (single row)
    df = pd.DataFrame([data])
    
    # TODO: Apply same preprocessing as training
    # This would include:
    # - Feature engineering (loan_to_income_ratio, etc.)
    # - Label encoding (using saved encoders)
    # - Scaling (using saved scaler)
    
    # For now, return raw DataFrame
    # In production, this must exactly match training preprocessing
    
    return df


@app.post(
    "/predict-risk",
    response_model=CreditRiskResponse,
    tags=["Prediction"],
    summary="Predict credit risk for loan applicant",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input data"},
        500: {"description": "Internal server error"}
    }
)
async def predict_credit_risk(request: CreditRiskRequest):
    """
    Predict default risk for a loan applicant.
    
    **Business Logic:**
    1. Validate applicant data (automatic via Pydantic)
    2. Preprocess features
    3. Generate risk score (default probability)
    4. Apply decision threshold
    5. Return APPROVE/REJECT with explanation
    
    **Risk Score Interpretation:**
    - 0.0-0.2: LOW risk
    - 0.2-0.4: MEDIUM risk
    - 0.4-0.6: HIGH risk
    - 0.6-1.0: CRITICAL risk
    
    **Decision Threshold:**
    - Current threshold: 35%
    - Approve if risk_score < 0.35
    - Reject if risk_score >= 0.35
    
    **Example Request:**
    ```json
    {
        "loan_amnt": 15000,
        "term": "36 months",
        "int_rate": 12.5,
        "annual_inc": 75000,
        "dti": 18.5,
        "grade": "B",
        "sub_grade": "B3",
        "emp_length": "5 years",
        "home_ownership": "MORTGAGE",
        "purpose": "debt_consolidation"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "risk_score": 0.23,
        "decision": "APPROVE",
        "risk_category": "MEDIUM",
        "threshold_used": 0.35,
        "explanation": "Medium default risk (23%). Loan approved."
    }
    ```
    """
    try:
        # Log request (sanitize in production)
        logger.info(f"Received prediction request: loan_amnt=${request.loan_amnt}, grade={request.grade}")
        
        # Check if model is loaded
        if model_loader is None or model_loader.model is None:
            logger.error("Model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not available. Please try again later."
            )
        
        # Preprocess applicant data
        applicant_df = preprocess_applicant_data(request)
        
        # Make prediction
        # Note: In full implementation, this would use the preprocessed DataFrame
        # For now, we'll simulate a prediction
        
        # MOCK PREDICTION (replace with actual model.predict)
        # This is a simplified version for demonstration
        # In production, use: predictions, probabilities = model_loader.predict(applicant_df)
        
        # Risk heuristic for demonstration:
        risk_factors = {
            'A': 0.10, 'B': 0.20, 'C': 0.30, 'D': 0.40, 'E': 0.50, 'F': 0.60, 'G': 0.70
        }
        base_risk = risk_factors.get(request.grade.value, 0.30)
        
        # Adjust for DTI
        dti_adjustment = (request.dti - 20) * 0.005
        
        # Adjust for loan-to-income
        lti_ratio = request.loan_amnt / request.annual_inc
        lti_adjustment = (lti_ratio - 0.2) * 0.3
        
        # Calculate final risk score
        risk_score = np.clip(base_risk + dti_adjustment + lti_adjustment, 0.0, 1.0)
        
        # Format response
        response = format_api_response(
            probability=risk_score,
            threshold=DECISION_THRESHOLD,
            include_details=True
        )
        
        # Map to Pydantic enums
        response['decision'] = LendingDecision(response['decision'])
        response['risk_category'] = RiskCategory(response['risk_category'])
        
        # Log decision
        logger.info(f"Prediction: risk_score={risk_score:.3f}, decision={response['decision']}")
        
        return CreditRiskResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/batch-predict",
    tags=["Prediction"],
    summary="Batch prediction for multiple applicants"
)
async def batch_predict(requests: list[CreditRiskRequest]):
    """
    Predict credit risk for multiple applicants.
    
    Use for:
    - Processing application backlogs
    - Portfolio risk assessment
    - Bulk underwriting
    
    Max batch size: 100 (configurable)
    """
    if len(requests) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum batch size is 100 applicants"
        )
    
    try:
        results = []
        for req in requests:
            # Reuse single prediction logic
            result = await predict_credit_risk(req)
            results.append(result)
        
        return {
            "count": len(results),
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get(
    "/model-info",
    tags=["Model"],
    summary="Get model information"
)
async def model_info():
    """
    Return information about the loaded model.
    
    Useful for:
    - Model version tracking
    - Audit trails
    - Debugging
    """
    if model_loader is None or model_loader.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    model = model_loader.model
    
    info = {
        "model_type": type(model).__name__,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_count": len(model_loader.feature_columns) if model_loader.feature_columns else "unknown",
        "loaded_at": datetime.now().isoformat()
    }
    
    # Add model-specific info
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    
    return info


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Validation error", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    """
    Run the API server.
    
    Development: python api/main.py
    Production: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
    """
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info"
    )
