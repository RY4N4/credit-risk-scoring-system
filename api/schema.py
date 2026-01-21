"""
Pydantic Schemas for Credit Risk API
------------------------------------
Request/response validation for production FastAPI.

Why Pydantic:
- Automatic input validation
- Type safety
- API documentation
- Error messages
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum


class LoanTerm(str, Enum):
    """Loan term options."""
    TERM_36 = "36 months"
    TERM_60 = "60 months"


class HomeOwnership(str, Enum):
    """Home ownership status options."""
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanPurpose(str, Enum):
    """Loan purpose categories."""
    DEBT_CONSOLIDATION = "debt_consolidation"
    CREDIT_CARD = "credit_card"
    HOME_IMPROVEMENT = "home_improvement"
    MAJOR_PURCHASE = "major_purchase"
    SMALL_BUSINESS = "small_business"
    MEDICAL = "medical"
    MOVING = "moving"
    VACATION = "vacation"
    HOUSE = "house"
    CAR = "car"
    WEDDING = "wedding"
    RENEWABLE_ENERGY = "renewable_energy"
    OTHER = "other"


class LoanGrade(str, Enum):
    """Lending Club loan grades."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class CreditRiskRequest(BaseModel):
    """
    Loan applicant data for credit risk prediction.
    
    All fields required for production inference.
    Values must match training data distribution.
    """
    
    loan_amnt: float = Field(
        ...,
        description="Loan amount requested ($)",
        ge=1000,
        le=40000,
        example=15000
    )
    
    term: LoanTerm = Field(
        ...,
        description="Loan term duration",
        example="36 months"
    )
    
    int_rate: float = Field(
        ...,
        description="Interest rate (%)",
        ge=5.0,
        le=30.0,
        example=12.5
    )
    
    annual_inc: float = Field(
        ...,
        description="Annual income ($)",
        ge=10000,
        le=500000,
        example=75000
    )
    
    dti: float = Field(
        ...,
        description="Debt-to-income ratio (%)",
        ge=0,
        le=50,
        example=18.5
    )
    
    grade: LoanGrade = Field(
        ...,
        description="Lending Club assigned grade",
        example="B"
    )
    
    sub_grade: str = Field(
        ...,
        description="Lending Club sub-grade (A1-G5)",
        example="B3"
    )
    
    emp_length: str = Field(
        ...,
        description="Employment length",
        example="5 years"
    )
    
    home_ownership: HomeOwnership = Field(
        ...,
        description="Home ownership status",
        example="MORTGAGE"
    )
    
    purpose: LoanPurpose = Field(
        ...,
        description="Loan purpose",
        example="debt_consolidation"
    )
    
    @validator('loan_amnt')
    def validate_loan_amount(cls, v):
        """Ensure loan amount is reasonable."""
        if v < 1000:
            raise ValueError("Minimum loan amount is $1,000")
        if v > 40000:
            raise ValueError("Maximum loan amount is $40,000")
        return v
    
    @validator('annual_inc')
    def validate_income(cls, v):
        """Ensure income is reasonable."""
        if v < 10000:
            raise ValueError("Minimum annual income is $10,000")
        return v
    
    @validator('dti')
    def validate_dti(cls, v):
        """Ensure DTI is within acceptable range."""
        if v < 0:
            raise ValueError("DTI cannot be negative")
        if v > 50:
            raise ValueError("DTI exceeds maximum threshold (50%)")
        return v
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class RiskCategory(str, Enum):
    """Risk level categories."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class LendingDecision(str, Enum):
    """Lending decision outcomes."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class CreditRiskResponse(BaseModel):
    """
    Credit risk prediction response.
    
    Returns risk score, decision, and explanation.
    """
    
    risk_score: float = Field(
        ...,
        description="Default probability (0-1)",
        ge=0.0,
        le=1.0,
        example=0.23
    )
    
    decision: LendingDecision = Field(
        ...,
        description="Lending decision (APPROVE/REJECT)",
        example="APPROVE"
    )
    
    risk_category: Optional[RiskCategory] = Field(
        None,
        description="Risk level category",
        example="MEDIUM"
    )
    
    threshold_used: Optional[float] = Field(
        None,
        description="Decision threshold applied",
        example=0.35
    )
    
    explanation: Optional[str] = Field(
        None,
        description="Human-readable decision explanation",
        example="Medium default risk (23%). Loan approved with standard terms."
    )
    
    class Config:
        schema_extra = {
            "example": {
                "risk_score": 0.23,
                "decision": "APPROVE",
                "risk_category": "MEDIUM",
                "threshold_used": 0.35,
                "explanation": "Medium default risk (23%). Loan approved with standard terms."
            }
        }


class HealthResponse(BaseModel):
    """API health check response."""
    
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    version: str = Field(..., example="1.0.0")


class ErrorResponse(BaseModel):
    """Error response format."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid input",
                "detail": "loan_amnt must be between $1,000 and $40,000"
            }
        }
