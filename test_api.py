"""
API Test Script
--------------
Test the credit risk API with sample requests.
"""

import requests
import json


def test_health_check():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    
    response = requests.get("http://localhost:8000/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_predict_low_risk():
    """Test prediction with low-risk applicant."""
    print("\n" + "="*60)
    print("Testing Low-Risk Applicant")
    print("="*60)
    
    # High income, low DTI, good grade
    payload = {
        "loan_amnt": 10000,
        "term": "36 months",
        "int_rate": 8.5,
        "annual_inc": 95000,
        "dti": 12.0,
        "grade": "A",
        "sub_grade": "A2",
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "purpose": "home_improvement"
    }
    
    response = requests.post(
        "http://localhost:8000/predict-risk",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_predict_high_risk():
    """Test prediction with high-risk applicant."""
    print("\n" + "="*60)
    print("Testing High-Risk Applicant")
    print("="*60)
    
    # Low income, high DTI, poor grade
    payload = {
        "loan_amnt": 30000,
        "term": "60 months",
        "int_rate": 25.5,
        "annual_inc": 35000,
        "dti": 38.0,
        "grade": "F",
        "sub_grade": "F4",
        "emp_length": "< 1 year",
        "home_ownership": "RENT",
        "purpose": "debt_consolidation"
    }
    
    response = requests.post(
        "http://localhost:8000/predict-risk",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_predict_medium_risk():
    """Test prediction with medium-risk applicant."""
    print("\n" + "="*60)
    print("Testing Medium-Risk Applicant")
    print("="*60)
    
    # Average applicant
    payload = {
        "loan_amnt": 15000,
        "term": "36 months",
        "int_rate": 12.5,
        "annual_inc": 65000,
        "dti": 20.0,
        "grade": "B",
        "sub_grade": "B3",
        "emp_length": "5 years",
        "home_ownership": "MORTGAGE",
        "purpose": "credit_card"
    }
    
    response = requests.post(
        "http://localhost:8000/predict-risk",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_invalid_input():
    """Test API with invalid input."""
    print("\n" + "="*60)
    print("Testing Invalid Input (should fail)")
    print("="*60)
    
    # Invalid: loan amount too high
    payload = {
        "loan_amnt": 50000,  # Exceeds max
        "term": "36 months",
        "int_rate": 12.5,
        "annual_inc": 65000,
        "dti": 20.0,
        "grade": "B",
        "sub_grade": "B3",
        "emp_length": "5 years",
        "home_ownership": "MORTGAGE",
        "purpose": "credit_card"
    }
    
    response = requests.post(
        "http://localhost:8000/predict-risk",
        json=payload
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_model_info():
    """Test model info endpoint."""
    print("\n" + "="*60)
    print("Testing Model Info")
    print("="*60)
    
    response = requests.get("http://localhost:8000/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("="*60)
    print("CREDIT RISK API TEST SUITE")
    print("="*60)
    print("\nMake sure API is running: python api/main.py")
    print("API should be available at: http://localhost:8000")
    
    input("\nPress Enter to start tests...")
    
    try:
        # Run tests
        test_health_check()
        test_model_info()
        test_predict_low_risk()
        test_predict_medium_risk()
        test_predict_high_risk()
        test_invalid_input()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETE")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure API is running: python api/main.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
