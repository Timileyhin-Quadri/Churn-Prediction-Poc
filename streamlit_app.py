import streamlit as st
import requests
import json

# ---------------------------
# Configuration
# ---------------------------
API_URL = "http://churnpoc-api.westeurope.azurecontainer.io:8000"  # Change if API is deployed elsewhere

# ---------------------------
# UI Setup
# ---------------------------
st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered")
st.title("📊 Customer Churn Prediction")
st.markdown("Enter customer details below to predict churn probability.")

# Health check
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.success("✅ API is running")
    else:
        st.warning("⚠️ API health check returned a non-200 status")
except Exception as e:
    st.error("❌ Could not connect to API. Make sure FastAPI is running.")
    st.stop()

# ---------------------------
# Input Form
# ---------------------------
with st.form("churn_form"):
    st.subheader("Customer Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["male", "female"])
    vintage = st.number_input("Vintage (months with company)", min_value=0, value=24)

    st.subheader("Account Balances")
    current_balance = st.number_input("Current Balance", value=5000.0)
    previous_month_end_balance = st.number_input("Previous Month End Balance", value=4500.0)
    average_monthly_balance_prevq = st.number_input("Average Monthly Balance Prev Q", value=4800.0)
    average_monthly_balance_prevq2 = st.number_input("Average Monthly Balance Prev Q-2", value=4600.0)
    current_month_balance = st.number_input("Current Month Balance", value=5000.0)
    previous_month_balance = st.number_input("Previous Month Balance", value=4500.0)

    st.subheader("Transactions")
    current_month_credit = st.number_input("Current Month Credit", value=3000.0)
    current_month_debit = st.number_input("Current Month Debit", value=2500.0)
    previous_month_credit = st.number_input("Previous Month Credit", value=2800.0)
    previous_month_debit = st.number_input("Previous Month Debit", value=2200.0)

    st.subheader("Other Info")
    dependents = st.number_input("Dependents", min_value=0, value=2)
    occupation = st.selectbox("Occupation", ["salaried", "self-employed", "student", "retired", "unknown"])
    customer_nw_category = st.number_input("Customer NW Category", min_value=1, max_value=3, value=2)
    city = st.number_input("City Code", min_value=1, value=101)
    branch_code = st.number_input("Branch Code", min_value=1, value=1001)

    submitted = st.form_submit_button("🔮 Predict Churn")

# ---------------------------
# Call API and Display Results
# ---------------------------
if submitted:
    payload = {
        "age": age,
        "gender": gender,
        "vintage": vintage,
        "current_balance": current_balance,
        "previous_month_end_balance": previous_month_end_balance,
        "average_monthly_balance_prevq": average_monthly_balance_prevq,
        "average_monthly_balance_prevq2": average_monthly_balance_prevq2,
        "current_month_balance": current_month_balance,
        "previous_month_balance": previous_month_balance,
        "current_month_credit": current_month_credit,
        "current_month_debit": current_month_debit,
        "previous_month_credit": previous_month_credit,
        "previous_month_debit": previous_month_debit,
        "dependents": dependents,
        "occupation": occupation,
        "customer_nw_category": customer_nw_category,
        "city": city,
        "branch_code": branch_code
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Prediction successful")

            st.metric("Churn Probability", f"{result['churn_probability']*100:.2f}%")
            st.metric("Prediction", result['churn_prediction'])
            st.metric("Risk Level", result['risk_level'])
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")

            st.subheader("Key Risk Factors")
            for factor in result["key_factors"]:
                st.write(f"- {factor}")
        else:
            st.error(f"❌ API returned error: {response.text}")
    except Exception as e:
        st.error(f"❌ Failed to connect to API: {e}")
