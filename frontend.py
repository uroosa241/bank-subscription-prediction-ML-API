

import streamlit as st
import requests

# Page title
st.title(" Bank Subscription Predictor")

st.write(
    "Enter the customer's information below to predict "
    "whether the customer will subscribe."
)

# -------------------------
# Customer information
# -------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

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
    value=1200.0
)

housing = st.selectbox(
    "Housing Loan",
    ["yes", "no"]
)

loan = st.selectbox(
    "Personal Loan",
    ["yes", "no"]
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
        "jan", "feb", "mar", "apr",
        "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec"
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
    ["unknown", "failure", "other", "success"]
)

# -------------------------
# Prediction button
# -------------------------

if st.button("🔮 Predict"):

    # Put all customer information into one dictionary
    customer = {
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

    # Send data to FastAPI
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=customer
    )

    # If API works
    if response.status_code == 200:

        result = response.json()

        prediction = result["prediction"]
        probability = result["probability_of_yes"]

        st.success(
            f"Prediction: {prediction.upper()}"
        )

        st.info(
            f"Probability of subscription: "
            f"{probability * 100:.2f}%"
        )

    else:

        st.error(
            "There was an error connecting to the prediction API."
        )