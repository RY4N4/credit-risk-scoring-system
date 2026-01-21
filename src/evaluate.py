"""
Model Evaluation Module for Credit Risk Scoring
-----------------------------------------------
Comprehensive evaluation with visualizations and business metrics.

Business Context:
- Beyond accuracy: precision, recall, AUC matter
- Cost-sensitive evaluation (FN more costly than FP)
- Visualizations for stakeholder communication
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
import os


class CreditModelEvaluator:
    """
    Advanced model evaluation with business-focused metrics.
    
    Focus Areas:
    - Discrimination ability (ROC-AUC)
    - Precision-Recall trade-offs
    - Cost-sensitive metrics
    - Visual reporting for non-technical stakeholders
    """
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_roc_curve(
        self,
        y_test: pd.Series,
        y_pred_proba_dict: Dict[str, np.ndarray],
        save_path: str = None
    ):
        """
        Plot ROC curves for multiple models.
        
        ROC Curve Interpretation:
        - X-axis: False Positive Rate (rejected good loans)
        - Y-axis: True Positive Rate (caught defaults)
        - Diagonal line: Random guessing
        - AUC = 0.5: No better than random
        - AUC > 0.7: Acceptable
        - AUC > 0.8: Good
        - AUC > 0.9: Excellent
        
        Args:
            y_test: True labels
            y_pred_proba_dict: {model_name: probabilities}
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, y_pred_proba in y_pred_proba_dict.items():
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{model_name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Rejected Good Loans)', fontsize=12)
        plt.ylabel('True Positive Rate (Caught Defaults)', fontsize=12)
        plt.title('ROC Curve - Credit Risk Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved ROC curve to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_test: pd.Series,
        y_pred_proba_dict: Dict[str, np.ndarray],
        save_path: str = None
    ):
        """
        Plot Precision-Recall curves.
        
        WHY THIS MATTERS FOR CREDIT RISK:
        - More informative than ROC for imbalanced classes
        - Shows trade-off between precision and recall
        - High precision: Few false alarms (approved loans won't default)
        - High recall: Catch most defaults (but may reject good loans)
        
        Business Trade-off:
        - Conservative lending: High precision, low recall
        - Aggressive lending: Lower precision, higher recall
        
        Args:
            y_test: True labels
            y_pred_proba_dict: {model_name: probabilities}
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, y_pred_proba in y_pred_proba_dict.items():
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            plt.plot(
                recall, precision,
                label=f'{model_name}',
                linewidth=2
            )
        
        # Baseline (no skill classifier)
        baseline = (y_test == 1).sum() / len(y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Baseline (Default Rate: {baseline:.2%})', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (% of Defaults Caught)', fontsize=12)
        plt.ylabel('Precision (% Approved that Don\'t Default)', fontsize=12)
        plt.title('Precision-Recall Curve - Credit Risk Models', fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved Precision-Recall curve to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        model_name: str,
        save_path: str = None
    ):
        """
        Visualize confusion matrix with business context.
        
        Confusion Matrix Layout:
                    Predicted
                Non-Def  Default
        Actual Non   TN      FP     <- Type I Error (lost revenue)
        Actual Def   FN      TP     <- Type II Error (loan loss)
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            model_name: Model name for title
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            xticklabels=['Non-Default', 'Default'],
            yticklabels=['Non-Default', 'Default']
        )
        
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        # Add annotations with business context
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate rates
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        plt.text(
            0.5, -0.15,
            f'FNR (Missed Defaults): {fnr:.2%} | FPR (Rejected Good): {fpr:.2%}',
            ha='center',
            transform=plt.gca().transAxes,
            fontsize=10
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        model,
        feature_names: list,
        model_name: str,
        top_n: int = 15,
        save_path: str = None
    ):
        """
        Plot feature importance for tree-based models.
        
        Helps answer:
        - Which features drive predictions?
        - Are we using relevant credit signals?
        - Can we simplify the model?
        
        Args:
            model: Trained model (must have feature_importances_)
            feature_names: List of feature names
            model_name: Model name for title
            top_n: Number of top features to display
            save_path: Path to save plot
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️  {model_name} doesn't support feature importance")
            return
        
        # Get feature importance
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        
        plt.barh(
            range(top_n),
            importance[indices],
            color='steelblue'
        )
        
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_n} Features - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved feature importance to {save_path}")
        
        plt.close()
    
    def calculate_business_metrics(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        avg_loan_amount: float = 15000,
        default_loss_rate: float = 0.6,
        interest_rate: float = 0.12,
        loan_term_years: float = 3
    ) -> Dict:
        """
        Calculate business impact metrics.
        
        BUSINESS CONTEXT:
        - Average loan: $15,000
        - Default loss rate: 60% of principal (after recovery)
        - Average interest rate: 12% APR
        - Average term: 3 years
        
        Metrics:
        - Expected loss from false negatives
        - Expected lost revenue from false positives
        - Net business impact
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            avg_loan_amount: Average loan size
            default_loss_rate: % of principal lost on default
            interest_rate: Annual interest rate
            loan_term_years: Loan duration
            
        Returns:
            Dictionary of business metrics
        """
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # False Negative: Approved a loan that defaulted
        loss_per_fn = avg_loan_amount * default_loss_rate  # $9,000 per default
        total_fn_loss = fn * loss_per_fn
        
        # False Positive: Rejected a good loan (lost interest revenue)
        interest_revenue = avg_loan_amount * interest_rate * loan_term_years  # $5,400
        total_fp_loss = fp * interest_revenue
        
        # True Positive: Correctly rejected bad loan (saved loss)
        total_saved = tp * loss_per_fn
        
        # True Negative: Correctly approved good loan (earned interest)
        total_earned = tn * interest_revenue
        
        # Net impact
        net_impact = total_earned - total_fn_loss - total_fp_loss
        
        print("\n" + "="*60)
        print("BUSINESS IMPACT ANALYSIS")
        print("="*60)
        print(f"\n💰 Revenue from Approved Good Loans (TN): ${total_earned:,.0f}")
        print(f"💸 Loss from Approved Bad Loans (FN): -${total_fn_loss:,.0f}")
        print(f"📉 Lost Revenue from Rejected Good Loans (FP): -${total_fp_loss:,.0f}")
        print(f"✅ Saved by Rejecting Bad Loans (TP): ${total_saved:,.0f}")
        print(f"\n{'─'*60}")
        print(f"🎯 NET BUSINESS IMPACT: ${net_impact:,.0f}")
        print(f"{'─'*60}")
        
        return {
            'total_earned': total_earned,
            'total_fn_loss': total_fn_loss,
            'total_fp_loss': total_fp_loss,
            'total_saved': total_saved,
            'net_impact': net_impact,
            'avg_loss_per_fn': loss_per_fn,
            'avg_loss_per_fp': interest_revenue
        }
    
    def generate_evaluation_report(
        self,
        y_test: pd.Series,
        models_dict: Dict,
        X_test: pd.DataFrame,
        feature_names: list
    ):
        """
        Generate comprehensive evaluation report with all visualizations.
        
        Args:
            y_test: True labels
            models_dict: {model_name: model}
            X_test: Test features
            feature_names: List of feature names
        """
        print("\n" + "="*60)
        print("GENERATING EVALUATION REPORT")
        print("="*60)
        
        # Collect predictions
        y_pred_proba_dict = {}
        y_pred_dict = {}
        
        for model_name, model in models_dict.items():
            y_pred_proba_dict[model_name] = model.predict_proba(X_test)[:, 1]
            y_pred_dict[model_name] = model.predict(X_test)
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        # ROC Curve
        self.plot_roc_curve(
            y_test,
            y_pred_proba_dict,
            save_path=os.path.join(self.output_dir, 'roc_curve.png')
        )
        
        # Precision-Recall Curve
        self.plot_precision_recall_curve(
            y_test,
            y_pred_proba_dict,
            save_path=os.path.join(self.output_dir, 'precision_recall_curve.png')
        )
        
        # Confusion matrices for each model
        for model_name, y_pred in y_pred_dict.items():
            self.plot_confusion_matrix(
                y_test,
                y_pred,
                model_name,
                save_path=os.path.join(self.output_dir, f'confusion_matrix_{model_name}.png')
            )
        
        # Feature importance for tree-based models
        for model_name, model in models_dict.items():
            if hasattr(model, 'feature_importances_'):
                self.plot_feature_importance(
                    model,
                    feature_names,
                    model_name,
                    save_path=os.path.join(self.output_dir, f'feature_importance_{model_name}.png')
                )
        
        print(f"\n✅ Evaluation report saved to {self.output_dir}/")
        print("   - roc_curve.png")
        print("   - precision_recall_curve.png")
        print("   - confusion_matrix_*.png")
        print("   - feature_importance_*.png")


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("-" * 60)
    print("\nProvides comprehensive evaluation including:")
    print("1. ROC-AUC curves")
    print("2. Precision-Recall curves")
    print("3. Confusion matrices")
    print("4. Feature importance")
    print("5. Business impact analysis")
