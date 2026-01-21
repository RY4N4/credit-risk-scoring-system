"""
Complete Training Pipeline
--------------------------
End-to-end training script that combines all modules.

Run this script to:
1. Load and process data
2. Engineer features
3. Train models
4. Evaluate performance
5. Save best model
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Tuple

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import CreditDataProcessor
from src.feature_engineering import CreditFeatureEngineering
from src.train import CreditRiskModelTrainer
from src.evaluate import CreditModelEvaluator


def main():
    """
    Complete training pipeline.
    """
    print("="*60)
    print("CREDIT RISK MODEL TRAINING PIPELINE")
    print("="*60)
    
    # ========================================
    # STEP 1: DATA PROCESSING
    # ========================================
    print("\n" + "="*60)
    print("STEP 1: DATA PROCESSING")
    print("="*60)
    
    processor = CreditDataProcessor()
    
    # Load raw data
    raw_data_path = 'data/raw/lending_club.csv'
    if not os.path.exists(raw_data_path):
        print(f"\n❌ Data file not found: {raw_data_path}")
        print("Please run: python generate_sample_data.py")
        return
    
    df = processor.load_data(raw_data_path)
    
    # Process data
    df_processed = processor.process_pipeline(df, fit=True)
    
    # ========================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================
    print("\n" + "="*60)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    feature_engineer = CreditFeatureEngineering()
    df_engineered = feature_engineer.create_all_features(df_processed)
    
    # ========================================
    # STEP 3: TRAIN/TEST SPLIT
    # ========================================
    print("\n" + "="*60)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("="*60)
    
    X_train, X_test, y_train, y_test = processor.split_data(df_engineered)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    train_data.to_csv('data/processed/train.csv', index=False)
    test_data.to_csv('data/processed/test.csv', index=False)
    print("✅ Saved processed data to data/processed/")
    
    # ========================================
    # STEP 4: MODEL TRAINING
    # ========================================
    print("\n" + "="*60)
    print("STEP 4: MODEL TRAINING")
    print("="*60)
    
    trainer = CreditRiskModelTrainer(random_state=42)
    
    # Handle class imbalance with SMOTE
    X_train_balanced, y_train_balanced = trainer.handle_class_imbalance(
        X_train, y_train, method='smote'
    )
    
    # Train Logistic Regression (baseline)
    lr_model = trainer.train_logistic_regression(
        X_train_balanced, y_train_balanced, use_class_weight=False  # Already balanced with SMOTE
    )
    
    # Train XGBoost (production model)
    xgb_model = trainer.train_xgboost(
        X_train_balanced, y_train_balanced, use_scale_pos_weight=False  # Already balanced
    )
    
    # ========================================
    # STEP 5: MODEL EVALUATION
    # ========================================
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)
    
    # Evaluate Logistic Regression
    lr_metrics = trainer.evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    
    # Evaluate XGBoost
    xgb_metrics = trainer.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    
    # Compare models
    trainer.compare_models()
    
    # ========================================
    # STEP 6: THRESHOLD OPTIMIZATION
    # ========================================
    print("\n" + "="*60)
    print("STEP 6: THRESHOLD OPTIMIZATION")
    print("="*60)
    
    optimal_threshold = trainer.find_optimal_threshold(
        xgb_model, X_test, y_test,
        cost_fp=1.0,  # Lost interest revenue: ~$1K
        cost_fn=5.0   # Default loss: ~$5K
    )
    
    # ========================================
    # STEP 7: GENERATE VISUALIZATIONS
    # ========================================
    print("\n" + "="*60)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("="*60)
    
    evaluator = CreditModelEvaluator(output_dir='reports')
    
    models_dict = {
        'Logistic Regression': lr_model,
        'XGBoost': xgb_model
    }
    
    evaluator.generate_evaluation_report(
        y_test,
        models_dict,
        X_test,
        feature_names=X_test.columns.tolist()
    )
    
    # ========================================
    # STEP 8: BUSINESS IMPACT ANALYSIS
    # ========================================
    print("\n" + "="*60)
    print("STEP 8: BUSINESS IMPACT ANALYSIS")
    print("="*60)
    
    # Use XGBoost predictions
    y_pred_xgb = xgb_model.predict(X_test)
    
    business_metrics = evaluator.calculate_business_metrics(
        y_test,
        y_pred_xgb,
        avg_loan_amount=15000,
        default_loss_rate=0.6,
        interest_rate=0.12,
        loan_term_years=3
    )
    
    # ========================================
    # STEP 9: SAVE MODELS AND ARTIFACTS
    # ========================================
    print("\n" + "="*60)
    print("STEP 9: SAVING MODELS AND ARTIFACTS")
    print("="*60)
    
    # Save preprocessing artifacts
    processor.save_artifacts('models')
    
    # Save models
    trainer.save_model('logistic_regression', 'models')
    trainer.save_model('xgboost', 'models')
    
    # Save best model as credit_model.pkl
    trainer.save_best_model('models', model_name='credit_model.pkl')
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*60)
    
    print("\n📊 Final Results:")
    print(f"   XGBoost ROC-AUC: {xgb_metrics['auc']:.4f}")
    print(f"   Optimal Threshold: {optimal_threshold:.3f}")
    print(f"   Business Impact: ${business_metrics['net_impact']:,.0f}")
    
    print("\n📁 Generated Files:")
    print("   ✅ models/credit_model.pkl (production model)")
    print("   ✅ models/scaler.pkl")
    print("   ✅ models/label_encoders.pkl")
    print("   ✅ data/processed/train.csv")
    print("   ✅ data/processed/test.csv")
    print("   ✅ reports/roc_curve.png")
    print("   ✅ reports/precision_recall_curve.png")
    print("   ✅ reports/confusion_matrix_*.png")
    print("   ✅ reports/feature_importance_*.png")
    
    print("\n🚀 Next Steps:")
    print("   1. Review visualizations in reports/")
    print("   2. Start API: python api/main.py")
    print("   3. Test API: http://localhost:8000/docs")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
