"""
Credit Risk Scoring - Interactive Web Application
=================================================

A Streamlit-based frontend for real-time credit risk assessment.
Users can input loan details and get instant predictions with visual feedback.

Author: ML Engineering Team
"""

import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint
API_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .approve-badge {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .reject-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .manual-review-badge {
        background-color: #ffc107;
        color: black;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running and healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_prediction(loan_data):
    """Get prediction from API."""
    try:
        response = requests.post(
            f"{API_URL}/predict-risk",
            json=loan_data,
            timeout=5
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Please ensure the API server is running."
    except Exception as e:
        return None, f"Error: {str(e)}"


def create_risk_gauge(risk_score):
    """Create a gauge chart for risk score visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Risk %", 'font': {'size': 24}},
        delta={'reference': 35, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#d4edda'},
                {'range': [20, 40], 'color': '#fff3cd'},
                {'range': [40, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 35
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_decision_bar(decision, risk_category):
    """Create a horizontal bar showing decision confidence."""
    colors = {
        'APPROVE': '#28a745',
        'REJECT': '#dc3545',
        'MANUAL_REVIEW': '#ffc107'
    }
    
    risk_colors = {
        'LOW': '#28a745',
        'MEDIUM': '#ffc107',
        'HIGH': '#dc3545'
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[1],
        y=[decision],
        orientation='h',
        marker=dict(color=colors.get(decision, '#6c757d')),
        text=[f"{decision} - {risk_category} RISK"],
        textposition='inside',
        textfont=dict(size=16, color='white'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        height=100,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        paper_bgcolor="white",
    )
    
    return fig


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">💳 Credit Risk Scoring System</div>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ **API Server is not running!**")
        st.info("""
        Please start the API server first:
        ```bash
        ./start_api.sh
        ```
        Or:
        ```bash
        export DYLD_LIBRARY_PATH="/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH"
        python api/main.py
        ```
        """)
        st.stop()
    
    st.success("✅ Connected to API Server")
    
    # Sidebar - Application Info
    with st.sidebar:
        st.header("📊 About")
        st.write("""
        This application uses **XGBoost** machine learning model to predict 
        loan default risk in real-time.
        
        **Model Performance:**
        - ROC-AUC: 0.63
        - Optimal Threshold: 23.6%
        - Business Impact: $6.4M
        
        **Features:**
        - Real-time predictions
        - Interactive visualizations
        - Business explanations
        """)
        
        st.header("📈 Model Info")
        try:
            response = requests.get(f"{API_URL}/model-info", timeout=2)
            if response.status_code == 200:
                model_info = response.json()
                st.json(model_info)
        except:
            st.warning("Could not fetch model info")
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main content
    st.header("📝 Enter Loan Application Details")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loan Details")
        loan_amnt = st.number_input(
            "Loan Amount ($)",
            min_value=500,
            max_value=40000,
            value=10000,
            step=500,
            help="Amount of loan requested ($500 - $40,000)"
        )
        
        int_rate = st.slider(
            "Interest Rate (%)",
            min_value=5.0,
            max_value=25.0,
            value=12.5,
            step=0.5,
            help="Annual interest rate (5% - 25%)"
        )
        
        term = st.selectbox(
            "Loan Term",
            options=["36 months", "60 months"],
            help="Length of loan repayment period"
        )
        
        grade = st.selectbox(
            "Credit Grade",
            options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            index=1,
            help="Lending Club's credit grade (A=best, G=worst)"
        )
        
        sub_grade = st.selectbox(
            "Credit Sub-Grade",
            options=[f"{grade}{i}" for i in range(1, 6)],
            help="More granular credit assessment"
        )
        
        purpose = st.selectbox(
            "Loan Purpose",
            options=[
                'debt_consolidation',
                'credit_card',
                'home_improvement',
                'major_purchase',
                'small_business',
                'car',
                'medical',
                'moving',
                'vacation',
                'house',
                'wedding',
                'renewable_energy',
                'other'
            ],
            index=1,
            help="Reason for taking the loan"
        )
    
    with col2:
        st.subheader("Borrower Profile")
        annual_inc = st.number_input(
            "Annual Income ($)",
            min_value=10000,
            max_value=500000,
            value=50000,
            step=5000,
            help="Annual gross income"
        )
        
        dti = st.slider(
            "Debt-to-Income Ratio (%)",
            min_value=0.0,
            max_value=40.0,
            value=15.0,
            step=0.5,
            help="Monthly debt payments / Monthly gross income"
        )
        
        emp_length = st.selectbox(
            "Employment Length",
            options=[
                '< 1 year',
                '1 year',
                '2 years',
                '3 years',
                '4 years',
                '5 years',
                '6 years',
                '7 years',
                '8 years',
                '9 years',
                '10+ years'
            ],
            index=5,
            help="Years at current job"
        )
        
        home_ownership = st.selectbox(
            "Home Ownership",
            options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
            index=0,
            help="Current housing situation"
        )
    
    # Predict button
    st.markdown("---")
    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
    
    with col_pred2:
        predict_button = st.button(
            "🔍 Analyze Credit Risk",
            use_container_width=True,
            type="primary"
        )
    
    if predict_button:
        # Prepare data
        loan_data = {
            "loan_amnt": float(loan_amnt),
            "int_rate": float(int_rate),
            "annual_inc": float(annual_inc),
            "dti": float(dti),
            "term": term,
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "purpose": purpose
        }
        
        # Show loading
        with st.spinner("Analyzing credit risk..."):
            result, error = get_prediction(loan_data)
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.markdown("---")
            st.header("📊 Credit Risk Assessment Results")
            
            # Create three columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                risk_score = result['risk_score']
                st.metric(
                    label="Default Risk Score",
                    value=f"{risk_score:.1%}",
                    delta=f"{risk_score - 0.35:.1%}" if risk_score > 0.35 else f"{risk_score - 0.35:.1%}",
                    delta_color="inverse"
                )
            
            with metric_col2:
                decision = result['decision']
                decision_emoji = {
                    'APPROVE': '✅',
                    'REJECT': '❌',
                    'MANUAL_REVIEW': '⚠️'
                }
                st.metric(
                    label="Lending Decision",
                    value=f"{decision_emoji.get(decision, '❓')} {decision.replace('_', ' ')}"
                )
            
            with metric_col3:
                risk_category = result['risk_category']
                category_emoji = {
                    'LOW': '🟢',
                    'MEDIUM': '🟡',
                    'HIGH': '🔴'
                }
                st.metric(
                    label="Risk Category",
                    value=f"{category_emoji.get(risk_category, '⚪')} {risk_category}"
                )
            
            # Visual representations
            st.markdown("---")
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                st.plotly_chart(
                    create_risk_gauge(risk_score),
                    use_container_width=True
                )
            
            with vis_col2:
                st.plotly_chart(
                    create_decision_bar(decision, risk_category),
                    use_container_width=True
                )
                
                # Explanation
                st.info(f"**Explanation:** {result['explanation']}")
                
                # Additional details
                with st.expander("📋 Technical Details"):
                    st.write(f"**Threshold Used:** {result['threshold_used']:.2%}")
                    st.write(f"**Model:** XGBoost Classifier")
                    st.write(f"**Features Analyzed:** 17 credit-specific features")
                    st.json(loan_data)
            
            # Recommendation section
            st.markdown("---")
            st.header("💡 Recommendations")
            
            if decision == 'APPROVE':
                st.success("""
                **✅ APPROVED**
                
                This loan application has been approved with standard terms:
                - Low default risk
                - Strong borrower profile
                - Meets lending criteria
                
                **Next Steps:**
                1. Send approval notification to applicant
                2. Prepare loan documents
                3. Schedule disbursement
                """)
            elif decision == 'REJECT':
                st.error("""
                **❌ REJECTED**
                
                This loan application has been declined:
                - High default risk
                - Does not meet lending criteria
                
                **Next Steps:**
                1. Send decline notification with reason
                2. Suggest credit improvement actions
                3. Offer to reconsider after 6 months
                """)
            else:
                st.warning("""
                **⚠️ MANUAL REVIEW REQUIRED**
                
                This application requires human review:
                - Moderate default risk
                - Additional assessment needed
                
                **Next Steps:**
                1. Assign to underwriting team
                2. Request additional documentation
                3. Conduct detailed risk assessment
                """)


if __name__ == "__main__":
    main()
