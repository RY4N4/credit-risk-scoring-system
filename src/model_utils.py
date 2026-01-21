"""
Model Utilities Module
---------------------
Helper functions for model loading, prediction, and inference.

Production Context:
- Safe model loading
- Input validation
- Prediction formatting
- Error handling
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import os


class ModelLoader:
    """
    Handles model loading and inference in production.
    
    Responsibilities:
    - Load trained models safely
    - Load preprocessing artifacts
    - Validate inputs
    - Make predictions
    """
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
    
    def load_model(self, model_name: str = 'credit_model.pkl'):
        """
        Load trained model.
        
        Args:
            model_name: Model filename
        """
        model_path = os.path.join(self.model_dir, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"✅ Loaded model from {model_path}")
    
    def load_preprocessing_artifacts(self):
        """
        Load scaler, encoders, and feature columns.
        """
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        features_path = os.path.join(self.model_dir, 'feature_columns.pkl')
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✅ Loaded scaler")
        
        if os.path.exists(encoders_path):
            self.label_encoders = joblib.load(encoders_path)
            print("✅ Loaded label encoders")
        
        if os.path.exists(features_path):
            self.feature_columns = joblib.load(features_path)
            print("✅ Loaded feature columns")
    
    def load_all(self, model_name: str = 'credit_model.pkl'):
        """
        Load model and all preprocessing artifacts.
        
        Args:
            model_name: Model filename
        """
        self.load_model(model_name)
        self.load_preprocessing_artifacts()
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            features: DataFrame with applicant features
            
        Returns:
            (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]
        
        return predictions, probabilities
    
    def predict_single(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict for a single applicant (API usage).
        
        Args:
            applicant_data: Dictionary with applicant features
            
        Returns:
            Dictionary with prediction and probability
        """
        # Convert to DataFrame
        df = pd.DataFrame([applicant_data])
        
        # Make prediction
        prediction, probability = self.predict(df)
        
        return {
            'prediction': int(prediction[0]),
            'default_probability': float(probability[0]),
            'risk_score': float(probability[0])
        }


def calculate_risk_score(probability: float) -> str:
    """
    Convert probability to risk category.
    
    Risk Categories:
    - LOW: < 20% default probability
    - MEDIUM: 20-40%
    - HIGH: 40-60%
    - CRITICAL: > 60%
    
    Args:
        probability: Default probability
        
    Returns:
        Risk category string
    """
    if probability < 0.20:
        return "LOW"
    elif probability < 0.40:
        return "MEDIUM"
    elif probability < 0.60:
        return "HIGH"
    else:
        return "CRITICAL"


def make_lending_decision(
    probability: float,
    threshold: float = 0.35
) -> str:
    """
    Make APPROVE/REJECT decision based on threshold.
    
    BUSINESS LOGIC:
    - Threshold = 0.35 means: Approve if default prob < 35%
    - Conservative threshold (0.25): Fewer defaults, more rejections
    - Aggressive threshold (0.45): More approvals, more defaults
    
    Typical thresholds:
    - Prime lending: 0.15-0.25
    - Subprime lending: 0.35-0.50
    
    Args:
        probability: Default probability
        threshold: Decision threshold
        
    Returns:
        "APPROVE" or "REJECT"
    """
    return "REJECT" if probability >= threshold else "APPROVE"


def format_api_response(
    probability: float,
    threshold: float = 0.35,
    include_details: bool = True
) -> Dict[str, Any]:
    """
    Format prediction as API response.
    
    Args:
        probability: Default probability
        threshold: Decision threshold
        include_details: Include risk category and explanation
        
    Returns:
        Formatted response dictionary
    """
    decision = make_lending_decision(probability, threshold)
    risk_category = calculate_risk_score(probability)
    
    response = {
        "risk_score": round(probability, 3),
        "decision": decision
    }
    
    if include_details:
        response["risk_category"] = risk_category
        response["threshold_used"] = threshold
        response["explanation"] = get_decision_explanation(probability, decision)
    
    return response


def get_decision_explanation(probability: float, decision: str) -> str:
    """
    Provide human-readable explanation for decision.
    
    Args:
        probability: Default probability
        decision: APPROVE or REJECT
        
    Returns:
        Explanation string
    """
    if decision == "APPROVE":
        return f"Low default risk ({probability:.1%}). Loan approved with standard terms."
    else:
        risk_level = calculate_risk_score(probability)
        return f"{risk_level} default risk ({probability:.1%}). Loan rejected or requires manual review."


if __name__ == "__main__":
    print("Model Utilities Module")
    print("-" * 60)
    print("\nProvides helper functions for:")
    print("1. Model loading")
    print("2. Making predictions")
    print("3. Formatting API responses")
    print("4. Risk categorization")
