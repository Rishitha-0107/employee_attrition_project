
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0f172a;
    }

    .stApp {
        background: linear-gradient(to right, #0f172a, #111827);
        color: white;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }

    .subtitle {
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 25px;
    }

    .card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(to right, #2563eb, #1d4ed8);
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }

    .risk-high {
        background: #7f1d1d;
        padding: 18px;
        border-radius: 14px;
        color: white;
        font-size: 22px;
        font-weight: bold;
    }

    .risk-medium {
        background: #78350f;
        padding: 18px;
        border-radius: 14px;
        color: white;
        font-size: 22px;
        font-weight: bold;
    }

    .risk-low {
        background: #14532d;
        padding: 18px;
        border-radius: 14px;
        color: white;
        font-size: 22px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("random_forest.pkl")

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚀 Employee Attrition Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict employee attrition risk using Machine Learning and HR Analytics.</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🧑 Employee Information")

st.sidebar.markdown("Enter employee details below")

Age = st.sidebar.slider("Age", 18, 60, 30)

MonthlyIncome = st.sidebar.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=50000,
    value=5000,
    step=500
)

JobSatisfaction = st.sidebar.slider(
    "Job Satisfaction",
    1,
    4,
    2
)

TotalWorkingYears = st.sidebar.slider(
    "Total Working Years",
    0,
    40,
    5
)

YearsAtCompany = st.sidebar.slider(
    "Years At Company",
    0,
    40,
    3
)

DistanceFromHome = st.sidebar.slider(
    "Distance From Home",
    1,
    30,
    5
)

EnvironmentSatisfaction = st.sidebar.slider(
    "Environment Satisfaction",
    1,
    4,
    2
)

WorkLifeBalance = st.sidebar.slider(
    "Work Life Balance",
    1,
    4,
    2
)

# =========================================================
# INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({
    'Age': [Age],
    'MonthlyIncome': [MonthlyIncome],
    'JobSatisfaction': [JobSatisfaction],
    'TotalWorkingYears': [TotalWorkingYears],
    'YearsAtCompany': [YearsAtCompany],
    'DistanceFromHome': [DistanceFromHome],
    'EnvironmentSatisfaction': [EnvironmentSatisfaction],
    'WorkLifeBalance': [WorkLifeBalance]
})

# =========================================================
# TOP METRICS
# =========================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.markdown(
        f'''
        <div class="metric-card">
            <h3>💰 Monthly Income</h3>
            <h2>${MonthlyIncome}</h2>
        </div>
        ''',
        unsafe_allow_html=True
    )

with metric2:
    st.markdown(
        f'''
        <div class="metric-card">
            <h3>😊 Job Satisfaction</h3>
            <h2>{JobSatisfaction}/4</h2>
        </div>
        ''',
        unsafe_allow_html=True
    )

with metric3:
    st.markdown(
        f'''
        <div class="metric-card">
            <h3>🏢 Experience</h3>
            <h2>{TotalWorkingYears} Years</h2>
        </div>
        ''',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# MAIN LAYOUT
# =========================================================

col1, col2 = st.columns([1, 1])

# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📋 Employee Input Summary")

    st.dataframe(input_data, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🤖 Prediction Dashboard")

    if st.button("Predict Attrition Risk"):

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)[0][1] * 100

        # ================================================
        # RISK CATEGORY
        # ================================================

        if probability >= 70:
            risk = "HIGH"

            st.markdown(
                f'''
                <div class="risk-high">
                ⚠️ HIGH ATTRITION RISK<br><br>
                Probability: {probability:.2f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

        elif probability >= 40:
            risk = "MEDIUM"

            st.markdown(
                f'''
                <div class="risk-medium">
                ⚠️ MEDIUM ATTRITION RISK<br><br>
                Probability: {probability:.2f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

        else:
            risk = "LOW"

            st.markdown(
                f'''
                <div class="risk-low">
                ✅ LOW ATTRITION RISK<br><br>
                Probability: {probability:.2f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ================================================
        # HR INSIGHTS
        # ================================================

        st.subheader("📌 HR Insights")

        insights = []

        if JobSatisfaction <= 2:
            insights.append("Low Job Satisfaction detected")

        if MonthlyIncome < 4000:
            insights.append("Salary may influence attrition")

        if WorkLifeBalance <= 2:
            insights.append("Poor work-life balance detected")

        if YearsAtCompany < 2:
            insights.append("New employee with higher transition risk")

        if DistanceFromHome > 15:
            insights.append("Long commute distance may affect retention")

        if len(insights) == 0:
            st.success("Employee profile appears stable")

        else:
            for item in insights:
                st.write(f"• {item}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ================================================
        # FINAL SUMMARY
        # ================================================

        st.subheader("📊 Prediction Summary")

        summary_df = pd.DataFrame({
            'Metric': [
                'Risk Level',
                'Attrition Probability'
            ],
            'Value': [
                risk,
                f'{probability:.2f}%'
            ]
        })

        st.table(summary_df)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    ### 📈 Machine Learning Models Used

    - Logistic Regression
    - Decision Tree
    - Random Forest
    - SVM
    - KNN
    - Naive Bayes
    - PCA
    - K-Means Clustering

)
