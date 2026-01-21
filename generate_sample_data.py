"""
Generate Sample Lending Club Dataset
------------------------------------
Creates a realistic synthetic dataset for testing and demonstration.

Use this if you don't have access to the actual Lending Club dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)


def generate_sample_lending_club_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic Lending Club-style data.
    
    Mimics real distributions and correlations.
    
    Args:
        n_samples: Number of loan records to generate
        
    Returns:
        DataFrame with synthetic loan data
    """
    print(f"Generating {n_samples} synthetic loan records...")
    
    # Loan amounts (typically $1K-$40K)
    loan_amnt = np.random.lognormal(mean=9.5, sigma=0.5, size=n_samples)
    loan_amnt = np.clip(loan_amnt, 1000, 40000).astype(int)
    
    # Loan term (36 or 60 months)
    term = np.random.choice([' 36 months', ' 60 months'], size=n_samples, p=[0.7, 0.3])
    
    # Interest rate (5-30%, correlated with grade)
    grades = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 
                             size=n_samples, 
                             p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.03, 0.02])
    
    # Interest rate based on grade
    grade_to_rate = {'A': (5, 10), 'B': (10, 15), 'C': (15, 20), 
                     'D': (20, 22), 'E': (22, 25), 'F': (25, 27), 'G': (27, 30)}
    
    int_rate = []
    sub_grade = []
    for grade in grades:
        low, high = grade_to_rate[grade]
        int_rate.append(np.random.uniform(low, high))
        sub_grade_num = random.randint(1, 5)
        sub_grade.append(f'{grade}{sub_grade_num}')
    
    int_rate = np.array(int_rate)
    
    # Annual income ($20K-$200K, log-normal)
    annual_inc = np.random.lognormal(mean=11.0, sigma=0.7, size=n_samples)
    annual_inc = np.clip(annual_inc, 20000, 300000).astype(int)
    
    # DTI (debt-to-income ratio, 0-40%)
    dti = np.random.beta(a=2, b=5, size=n_samples) * 40
    
    # Employment length
    emp_length_options = ['< 1 year', '1 year', '2 years', '3 years', '4 years',
                         '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
    emp_length = np.random.choice(emp_length_options, size=n_samples, 
                                 p=[0.05, 0.08, 0.10, 0.12, 0.10, 0.15, 0.12, 0.10, 0.08, 0.05, 0.05])
    
    # Home ownership
    home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
                                     size=n_samples,
                                     p=[0.40, 0.45, 0.12, 0.03])
    
    # Loan purpose
    purpose_options = ['debt_consolidation', 'credit_card', 'home_improvement',
                      'major_purchase', 'small_business', 'car', 'medical',
                      'moving', 'vacation', 'house', 'wedding', 'renewable_energy', 'other']
    purpose_probs = [0.30, 0.20, 0.15, 0.10, 0.05, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    purpose = np.random.choice(purpose_options, size=n_samples, p=purpose_probs)
    
    # Loan status (target variable)
    # Default probability based on grade and other factors
    default_prob = []
    for i in range(n_samples):
        # Base probability from grade
        grade_risk = {'A': 0.05, 'B': 0.10, 'C': 0.15, 'D': 0.20, 'E': 0.25, 'F': 0.30, 'G': 0.35}
        base_prob = grade_risk[grades[i]]
        
        # Adjust for DTI (higher DTI = higher risk)
        dti_factor = 1 + (dti[i] - 20) / 100
        
        # Adjust for loan-to-income
        lti_ratio = loan_amnt[i] / annual_inc[i]
        lti_factor = 1 + (lti_ratio - 0.2) * 0.5
        
        # Adjust for term (60 months riskier)
        term_factor = 1.2 if term[i] == ' 60 months' else 1.0
        
        # Calculate final probability
        prob = base_prob * dti_factor * lti_factor * term_factor
        prob = np.clip(prob, 0, 0.6)
        default_prob.append(prob)
    
    # Generate loan status based on probability
    loan_status = []
    for prob in default_prob:
        if np.random.random() < prob:
            # Default
            loan_status.append(np.random.choice(['Charged Off', 'Default'], p=[0.9, 0.1]))
        else:
            # Non-default
            loan_status.append(np.random.choice(['Fully Paid', 'Current'], p=[0.8, 0.2]))
    
    # Create DataFrame
    df = pd.DataFrame({
        'loan_amnt': loan_amnt,
        'term': term,
        'int_rate': int_rate,
        'grade': grades,
        'sub_grade': sub_grade,
        'emp_length': emp_length,
        'home_ownership': home_ownership,
        'annual_inc': annual_inc,
        'purpose': purpose,
        'dti': dti,
        'loan_status': loan_status
    })
    
    # Calculate statistics
    default_rate = df['loan_status'].isin(['Charged Off', 'Default']).mean()
    
    print(f"\n✅ Generated {len(df)} loan records")
    print(f"   Default rate: {default_rate:.2%}")
    print(f"   Grade distribution:")
    for grade in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        pct = (df['grade'] == grade).mean()
        print(f"      {grade}: {pct:.1%}")
    
    return df


def add_noise_and_missing_values(df: pd.DataFrame, missing_rate: float = 0.05) -> pd.DataFrame:
    """
    Add realistic missing values and noise.
    
    Args:
        df: Clean DataFrame
        missing_rate: Fraction of values to make missing
        
    Returns:
        DataFrame with missing values
    """
    df_noisy = df.copy()
    
    # Add missing values to specific columns (realistic patterns)
    missing_columns = ['emp_length', 'dti', 'annual_inc']
    
    for col in missing_columns:
        if col in df_noisy.columns:
            n_missing = int(len(df_noisy) * missing_rate)
            missing_indices = np.random.choice(df_noisy.index, size=n_missing, replace=False)
            df_noisy.loc[missing_indices, col] = np.nan
    
    print(f"\n✅ Added missing values to {len(missing_columns)} columns")
    
    return df_noisy


if __name__ == "__main__":
    import os
    
    print("="*60)
    print("SAMPLE DATA GENERATOR")
    print("="*60)
    
    # Generate data
    df = generate_sample_lending_club_data(n_samples=10000)
    
    # Add missing values
    df = add_noise_and_missing_values(df, missing_rate=0.05)
    
    # Save to data/raw
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'lending_club.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved sample data to: {output_path}")
    print("\nYou can now run:")
    print("  1. python src/data_processing.py")
    print("  2. python src/train.py")
    print("  3. python api/main.py")
