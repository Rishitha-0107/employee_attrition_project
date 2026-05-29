
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("random_forest.pkl")

# =========================
# TITLE
# =========================
st.title("🚀 AI-Powered Employee Attrition Prediction System")

st.markdown("---")

st.write(
    "Predict employee attrition risk using Machine Learning and HR analytics."
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Employee Information")

Age = st.sidebar.slider("Age", 18, 60, 30)

MonthlyIncome = st.sidebar.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=50000,
    value=5000
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

# =========================
# INPUT DATAFRAME
# =========================
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

# =========================
# MAIN CONTENT
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Employee Input Data")
    st.dataframe(input_data)

with col2:
    st.subheader("🤖 Prediction")

    if st.button("Predict Attrition Risk"):

        try:
            prediction = model.predict(input_data)

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_data)[0][1] * 100
            else:
                probability = 50

            if probability >= 70:
                risk = "HIGH"
                st.error(f"⚠️ High Attrition Risk: {probability:.2f}%")

            elif probability >= 40:
                risk = "MEDIUM"
                st.warning(f"⚠️ Medium Attrition Risk: {probability:.2f}%")

            else:
                risk = "LOW"
                st.success(f"✅ Low Attrition Risk: {probability:.2f}%")

            st.markdown("---")

            st.subheader("📌 HR Insights")

            if JobSatisfaction <= 2:
                st.write("- Low Job Satisfaction may increase attrition risk")

            if MonthlyIncome < 4000:
                st.write("- Low salary may influence employee resignation")

            if WorkLifeBalance <= 2:
                st.write("- Poor work-life balance detected")

            if YearsAtCompany < 2:
                st.write("- New employee with higher transition probability")

            st.markdown("---")

            st.subheader("📊 Prediction Summary")

            st.write(f"**Risk Level:** {risk}")
            st.write(f"**Attrition Probability:** {probability:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown(
    """
    ### 📈 Models Used

    - Logistic Regression
    - Decision Tree
    - Random Forest
    - SVM
    - KNN
    - Naive Bayes
    - PCA
    - K-Means Clustering

    ### ☁️ Deployment

    - Streamlit
    - AWS S3
    - SageMaker
    - Lambda
    - API Gateway
    """
)
