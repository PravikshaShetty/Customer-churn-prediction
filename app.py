import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a customer's details below to estimate their probability of churning."
)

available_models = {name: path for name, path in MODEL_FILES.items() if os.path.exists(path)}

if not available_models:
    st.error(
        "No trained model files found. Run the notebook "
        "(`Customer_Churn_Prediction.ipynb`) first — it saves "
        "`logistic_model.pkl`, `rf_model.pkl`, and `xgb_model.pkl` "
        "into this same folder."
    )
    st.stop()

model_name = st.selectbox("Model", list(available_models.keys()))
model = load_model(available_models[model_name])

st.subheader("Customer details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple lines", ["No", "No phone service", "Yes"])
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online security", ["No", "No internet service", "Yes"])

with col2:
    online_backup = st.selectbox("Online backup", ["No", "No internet service", "Yes"])
    device_protection = st.selectbox("Device protection", ["No", "No internet service", "Yes"])
    tech_support = st.selectbox("Tech support", ["No", "No internet service", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
    streaming_movies = st.selectbox("Streaming movies", ["No", "No internet service", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless billing", ["No", "Yes"])
    payment_method = st.selectbox(
        "Payment method",
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
    )

st.subheader("Billing details")
c1, c2, c3 = st.columns(3)
with c1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
with c2:
    monthly_charges = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0)
with c3:
    total_charges = st.number_input("Total charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))

if st.button("Predict churn risk", type="primary"):
    input_df = pd.DataFrame([{
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "tenure": tenure,
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
    }])

    proba = model.predict_proba(input_df)[0, 1]
    prediction = "Likely to churn" if proba >= 0.5 else "Likely to stay"

    st.metric("Churn probability", f"{proba:.1%}")
    if proba >= 0.5:
        st.error(f"⚠️ {prediction}")
    else:
        st.success(f"✅ {prediction}")

    st.progress(min(int(proba * 100), 100))

st.caption(
    f"Model in use: **{model_name}**. "
    "Trained on the IBM/Kaggle Telco Customer Churn dataset."
)
