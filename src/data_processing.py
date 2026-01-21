"""
Data Processing Module for Credit Risk Scoring
-----------------------------------------------
Handles data loading, cleaning, and preprocessing for credit risk modeling.

Business Context:
- Missing values can indicate unstable financial history
- Categorical features (loan grade, employment) are strong credit indicators
- Proper encoding and scaling ensure model stability
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


class CreditDataProcessor:
    """
    Handles all data preprocessing for credit risk modeling.
    
    Production Considerations:
    - Maintains consistency between training and inference
    - Saves preprocessing artifacts (scalers, encoders)
    - Validates data quality
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw Lending Club dataset.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Raw DataFrame
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant features for credit risk modeling.
        
        Feature Selection Rationale:
        - loan_amnt: Higher amounts = higher risk
        - int_rate: Lender's risk assessment signal
        - annual_inc: Income stability indicator
        - dti: Debt burden measure
        - grade/sub_grade: Lender's credit quality assessment
        - emp_length: Employment stability
        - home_ownership: Financial stability proxy
        - purpose: Loan intent (debt consolidation riskier than home improvement)
        - term: Loan duration affects default probability
        """
        
        # Core credit features
        feature_cols = [
            'loan_amnt',           # Loan amount requested
            'term',                # 36/60 months
            'int_rate',            # Interest rate (risk signal)
            'annual_inc',          # Annual income
            'dti',                 # Debt-to-income ratio
            'grade',               # Lending Club assigned grade (A-G)
            'sub_grade',           # Finer granularity (A1-G5)
            'emp_length',          # Employment length
            'home_ownership',      # RENT, OWN, MORTGAGE
            'purpose',             # Loan purpose
            'loan_status'          # Target variable
        ]
        
        # Filter to only include columns that exist
        available_cols = [col for col in feature_cols if col in df.columns]
        df_selected = df[available_cols].copy()
        
        print(f"Selected {len(available_cols)} features for modeling")
        return df_selected
    
    def define_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable: Default (1) vs Non-Default (0).
        
        Business Logic:
        - Charged Off = Loan defaulted, significant loss
        - Default = Explicit default
        - Fully Paid/Current = Non-default
        
        Args:
            df: DataFrame with loan_status column
            
        Returns:
            DataFrame with 'default' binary target
        """
        # Define default statuses
        default_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']
        
        df['default'] = df['loan_status'].isin(default_statuses).astype(int)
        
        # Remove loan_status (target leakage prevention)
        df = df.drop('loan_status', axis=1)
        
        default_rate = df['default'].mean()
        print(f"Default rate: {default_rate:.2%}")
        print(f"Defaults: {df['default'].sum()}, Non-defaults: {(df['default']==0).sum()}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with business-appropriate strategies.
        
        Strategy:
        - Numerical: Median imputation (robust to outliers)
        - Categorical: Mode imputation or 'Unknown' category
        - Drop rows if >50% features missing (data quality issue)
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        print("\nHandling missing values...")
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'default' in numerical_cols:
            numerical_cols.remove('default')
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Store for later use
        self.numerical_columns = numerical_cols
        self.categorical_columns = categorical_cols
        
        # Numerical: median imputation
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  Imputed {col} with median: {median_val:.2f}")
        
        # Categorical: mode or 'Unknown'
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"  Imputed {col} with: {mode_val}")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables for ML models.
        
        Strategy:
        - Label Encoding for ordinal features (grade: A < B < C)
        - One-Hot Encoding for nominal features (home_ownership)
        
        Args:
            df: DataFrame with categorical columns
            fit: If True, fit encoders; if False, use existing encoders
            
        Returns:
            DataFrame with encoded categorical variables
        """
        print("\nEncoding categorical variables...")
        
        # Ordinal encoding for grade (preserves ordering)
        if 'grade' in df.columns:
            grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
            df['grade'] = df['grade'].map(grade_mapping)
            print("  Encoded 'grade' with ordinal mapping")
        
        # Label encoding for other categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories gracefully
                    le = self.label_encoders[col]
                    df[col] = df[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )
                else:
                    raise ValueError(f"No encoder found for column: {col}")
            
            print(f"  Label encoded: {col}")
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features for model stability.
        
        Why scaling matters:
        - Logistic Regression sensitive to feature scales
        - XGBoost less sensitive but scaling doesn't hurt
        - Ensures features contribute proportionally
        
        Args:
            df: DataFrame with numerical features
            fit: If True, fit scaler; if False, use existing scaler
            
        Returns:
            DataFrame with scaled features
        """
        print("\nScaling numerical features...")
        
        # Separate target and features
        target = df['default'] if 'default' in df.columns else None
        features = df.drop('default', axis=1) if 'default' in df.columns else df
        
        if fit:
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        
        # Create scaled DataFrame
        scaled_df = pd.DataFrame(
            scaled_features,
            columns=features.columns,
            index=features.index
        )
        
        # Add target back
        if target is not None:
            scaled_df['default'] = target
        
        print(f"  Scaled {len(features.columns)} features")
        return scaled_df
    
    def process_pipeline(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Pipeline:
        1. Select features
        2. Define target
        3. Handle missing values
        4. Encode categoricals
        5. Scale numericals
        
        Args:
            df: Raw DataFrame
            fit: If True, fit transformers; if False, transform only
            
        Returns:
            Processed DataFrame ready for modeling
        """
        if fit:
            df = self.select_features(df)
            df = self.define_target(df)
        
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df, fit=fit)
        df = self.scale_features(df, fit=fit)
        
        if fit:
            self.feature_columns = [col for col in df.columns if col != 'default']
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test sets with stratification.
        
        Stratification ensures both sets have similar default rates.
        
        Args:
            df: Processed DataFrame
            test_size: Fraction for test set
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop('default', axis=1)
        y = df['default']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Maintain default rate in both sets
        )
        
        print(f"\nTrain set: {len(X_train)} samples (default rate: {y_train.mean():.2%})")
        print(f"Test set: {len(X_test)} samples (default rate: {y_test.mean():.2%})")
        
        return X_train, X_test, y_train, y_test
    
    def save_artifacts(self, output_dir: str):
        """
        Save preprocessing artifacts for production deployment.
        
        Critical for inference consistency.
        
        Args:
            output_dir: Directory to save artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(output_dir, 'feature_columns.pkl'))
        
        print(f"\nSaved preprocessing artifacts to {output_dir}")
    
    def load_artifacts(self, input_dir: str):
        """
        Load preprocessing artifacts for inference.
        
        Args:
            input_dir: Directory containing artifacts
        """
        self.scaler = joblib.load(os.path.join(input_dir, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(input_dir, 'label_encoders.pkl'))
        self.feature_columns = joblib.load(os.path.join(input_dir, 'feature_columns.pkl'))
        
        print(f"Loaded preprocessing artifacts from {input_dir}")


if __name__ == "__main__":
    # Example usage
    processor = CreditDataProcessor()
    
    # Load and process data
    df = processor.load_data('data/raw/lending_club.csv')
    df_processed = processor.process_pipeline(df, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(df_processed)
    
    # Save artifacts
    processor.save_artifacts('models')
    
    # Save processed data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('data/processed/train.csv', index=False)
    test_data.to_csv('data/processed/test.csv', index=False)
    
    print("\n✅ Data processing complete!")
