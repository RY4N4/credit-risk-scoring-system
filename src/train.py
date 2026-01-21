"""
Model Training Module for Credit Risk Scoring
---------------------------------------------
Trains and compares Logistic Regression and XGBoost models.

Business Context:
- Class imbalance is critical (defaults are minority class)
- False negatives (approving bad loans) are costly
- False positives (rejecting good loans) lose revenue
- Model must be explainable for regulatory compliance
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import joblib
import os

# ML libraries
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


class CreditRiskModelTrainer:
    """
    Trains credit risk models with focus on production readiness.
    
    Key Considerations:
    - Handles class imbalance (SMOTE + class weights)
    - Tracks multiple metrics (AUC, Precision, Recall)
    - Saves models and training artifacts
    - Provides model comparison framework
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.training_history = {}
        
    def handle_class_imbalance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = 'smote'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Address class imbalance in training data.
        
        WHY THIS MATTERS:
        - Default rate typically 10-20% (imbalanced)
        - Models biased toward majority class (predict all non-default)
        - SMOTE creates synthetic minority samples
        
        Methods:
        - 'smote': Synthetic Minority Over-sampling
        - 'none': No resampling (use class weights in model)
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: Resampling method
            
        Returns:
            Resampled X_train, y_train
        """
        original_default_rate = y_train.mean()
        print(f"\nOriginal default rate: {original_default_rate:.2%}")
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            new_default_rate = y_resampled.mean()
            print(f"After SMOTE default rate: {new_default_rate:.2%}")
            print(f"Training samples: {len(X_train)} -> {len(X_resampled)}")
            
            return X_resampled, y_resampled
        else:
            print("Using class weights instead of resampling")
            return X_train, y_train
    
    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_class_weight: bool = True
    ) -> LogisticRegression:
        """
        Train Logistic Regression baseline model.
        
        WHY LOGISTIC REGRESSION:
        - Fast training and inference
        - Highly interpretable (coefficients = feature impact)
        - Industry standard for credit scoring
        - Good baseline for comparison
        
        Pros:
        - Interpretable coefficients
        - Probabilistic output
        - Regulatory friendly
        
        Cons:
        - Assumes linear relationships
        - Can't capture complex interactions
        - Limited by feature engineering quality
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_class_weight: Apply class weighting
            
        Returns:
            Trained LogisticRegression model
        """
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION (BASELINE)")
        print("="*60)
        
        # Configure model
        params = {
            'max_iter': 1000,
            'random_state': self.random_state,
            'class_weight': 'balanced' if use_class_weight else None,
            'solver': 'lbfgs'  # Good for small-medium datasets
        }
        
        model = LogisticRegression(**params)
        
        print(f"Training with {len(X_train)} samples...")
        model.fit(X_train, y_train)
        
        # Store model
        self.models['logistic_regression'] = model
        
        print("✅ Logistic Regression trained successfully")
        
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_scale_pos_weight: bool = True
    ) -> XGBClassifier:
        """
        Train XGBoost model (final production model).
        
        WHY XGBOOST FOR CREDIT RISK:
        - Handles non-linear relationships
        - Captures feature interactions automatically
        - Robust to outliers
        - Built-in regularization prevents overfitting
        - Industry standard for tabular data
        
        Pros over Logistic Regression:
        - Better predictive performance (typically 5-10% AUC gain)
        - Automatic feature interaction detection
        - Handles missing values natively
        - Less feature engineering needed
        
        Cons:
        - Less interpretable (use SHAP for explanations)
        - Slower training
        - More hyperparameters to tune
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_scale_pos_weight: Apply positive class weighting
            
        Returns:
            Trained XGBClassifier model
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST (FINAL MODEL)")
        print("="*60)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio: {scale_pos_weight:.2f}")
        
        # Configure model with production-ready parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,                    # Control overfitting
            'learning_rate': 0.1,              # Standard learning rate
            'n_estimators': 100,               # Number of trees
            'subsample': 0.8,                  # Row sampling for robustness
            'colsample_bytree': 0.8,          # Column sampling
            'min_child_weight': 5,            # Minimum samples per leaf
            'gamma': 0.1,                      # Min loss reduction for split
            'reg_alpha': 0.1,                  # L1 regularization
            'reg_lambda': 1.0,                 # L2 regularization
            'scale_pos_weight': scale_pos_weight if use_scale_pos_weight else 1,
            'random_state': self.random_state,
            'eval_metric': 'auc'               # Optimize for AUC
        }
        
        model = XGBClassifier(**params)
        
        print(f"Training with {len(X_train)} samples...")
        model.fit(
            X_train, 
            y_train,
            verbose=False
        )
        
        # Store model
        self.models['xgboost'] = model
        
        print("✅ XGBoost trained successfully")
        
        return model
    
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict:
        """
        Comprehensive model evaluation.
        
        Metrics Tracked:
        - ROC-AUC: Overall discrimination ability
        - Precision: % of approved loans that don't default
        - Recall: % of defaults caught by model
        - F1-Score: Harmonic mean of precision and recall
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name for tracking
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # ROC-AUC Score
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"\n📊 ROC-AUC Score: {auc_score:.4f}")
        
        # Classification Report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        print(f"                Predicted")
        print(f"              Non-Default  Default")
        print(f"Actual Non-D     {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"Actual Default   {cm[1][0]:6d}    {cm[1][1]:6d}")
        
        # Calculate business metrics
        tn, fp, fn, tp = cm.ravel()
        
        # False Negative Rate (missed defaults - costly!)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"\n⚠️  False Negative Rate: {fnr:.2%} (approved bad loans)")
        
        # False Positive Rate (rejected good loans - lost revenue)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"⚠️  False Positive Rate: {fpr:.2%} (rejected good loans)")
        
        # Store metrics
        metrics = {
            'auc': auc_score,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'false_negative_rate': fnr,
            'false_positive_rate': fpr
        }
        
        self.training_history[model_name] = metrics
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models side-by-side.
        
        Returns:
            DataFrame with model comparison
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison = []
        
        for model_name, metrics in self.training_history.items():
            comparison.append({
                'Model': model_name,
                'ROC-AUC': f"{metrics['auc']:.4f}",
                'FNR (Missed Defaults)': f"{metrics['false_negative_rate']:.2%}",
                'FPR (Rejected Good)': f"{metrics['false_positive_rate']:.2%}"
            })
        
        df_comparison = pd.DataFrame(comparison)
        print("\n", df_comparison.to_string(index=False))
        
        return df_comparison
    
    def find_optimal_threshold(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cost_fp: float = 1.0,
        cost_fn: float = 5.0
    ) -> float:
        """
        Find optimal decision threshold based on business costs.
        
        BUSINESS LOGIC:
        - False Negative (approving bad loan): Lose principal + interest
          - Average loss: $5,000-$15,000 per default
        - False Positive (rejecting good loan): Lose interest revenue
          - Average loss: $500-$2,000 per rejection
        - Typical ratio: FN is 5-10x more costly than FP
        
        Default threshold (0.5) may not be optimal!
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            cost_fp: Cost of false positive (rejected good loan)
            cost_fn: Cost of false negative (approved bad loan)
            
        Returns:
            Optimal probability threshold
        """
        print("\n" + "="*60)
        print("FINDING OPTIMAL THRESHOLD")
        print("="*60)
        
        # Get prediction probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Calculate cost for each threshold
        min_cost = float('inf')
        optimal_threshold = 0.5
        
        for i, threshold in enumerate(thresholds):
            # Predictions at this threshold
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Total cost
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            
            if total_cost < min_cost:
                min_cost = total_cost
                optimal_threshold = threshold
        
        print(f"\nBusiness Cost Assumptions:")
        print(f"  False Positive cost: ${cost_fp:,.0f} (lost interest revenue)")
        print(f"  False Negative cost: ${cost_fn:,.0f} (default loss)")
        print(f"  Cost ratio (FN/FP): {cost_fn/cost_fp:.1f}x")
        
        print(f"\n📍 Default threshold: 0.50")
        print(f"🎯 Optimal threshold: {optimal_threshold:.3f}")
        print(f"💰 Expected cost reduction: {((0.5 - optimal_threshold)/0.5 * 100):.1f}%")
        
        return optimal_threshold
    
    def save_model(self, model_name: str, output_dir: str):
        """
        Save trained model for production deployment.
        
        Args:
            model_name: Name of model to save
            output_dir: Directory to save model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f'{model_name}.pkl')
        joblib.dump(self.models[model_name], model_path)
        
        print(f"\n✅ Saved {model_name} to {model_path}")
    
    def save_best_model(self, output_dir: str, model_name: str = 'credit_model.pkl'):
        """
        Save the best performing model (typically XGBoost).
        
        Args:
            output_dir: Directory to save model
            model_name: Filename for model
        """
        if 'xgboost' in self.models:
            best_model = self.models['xgboost']
            best_model_name = 'xgboost'
        elif 'logistic_regression' in self.models:
            best_model = self.models['logistic_regression']
            best_model_name = 'logistic_regression'
        else:
            raise ValueError("No models trained yet!")
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, model_name)
        joblib.dump(best_model, model_path)
        
        print(f"\n✅ Saved best model ({best_model_name}) to {model_path}")


if __name__ == "__main__":
    print("Model Training Module")
    print("-" * 60)
    print("\nThis module trains and compares:")
    print("1. Logistic Regression (baseline)")
    print("2. XGBoost (production model)")
    print("\nHandles class imbalance and finds optimal thresholds.")
