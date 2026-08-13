import streamlit as st
import requests

st.set_page_config(
    page_title="Bank Subscription Predictor",
    page_icon="",
    layout="centered"
)

st.title("Bank Subscription Predictor")
st.write("Enter customer information to predict whether the customer will subscribe.")

# Customer information
age = st.number_input("Age", min_value=18, max_value=100, value=30)

job = st.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ]
)

marital = st.selectbox(
    "Marital Status",
    ["married", "single", "divorced"]
)

education = st.selectbox(
    "Education",
    ["primary", "secondary", "tertiary", "unknown"]
)

default = st.selectbox(
    "Default",
    ["no", "yes"]
)

balance = st.number_input(
    "Balance",
    value=0.0
)

housing = st.selectbox(
    "Housing Loan",
    ["no", "yes"]
)

loan = st.selectbox(
    "Personal Loan",
    ["no", "yes"]
)

contact = st.selectbox(
    "Contact",
    ["cellular", "telephone", "unknown"]
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=15
)

month = st.selectbox(
    "Month",
    [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]
)

campaign = st.number_input(
    "Campaign",
    min_value=1,
    value=1
)

pdays = st.number_input(
    "Previous Contact Days",
    value=-1
)

previous = st.number_input(
    "Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.selectbox(
    "Previous Outcome",
    ["failure", "success", "unknown", "other"]
)


# Prediction button
if st.button("Predict", type="primary"):

    customer_data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=customer_data
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            probability = result["probability_of_yes"]

            if prediction == "yes":
                st.success("✅ Customer is likely to subscribe!")
            else:
                st.error("❌ Customer is unlikely to subscribe.")

            st.metric(
                "Probability of Subscription",
                f"{probability * 100:.2f}%"
            )

        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI is not running. "
            "Please start the FastAPI server first."
        )