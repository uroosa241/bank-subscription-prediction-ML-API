import streamlit as st
import joblib
import pandas as pd

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Subscription Predictor",
    page_icon="",
    layout="centered"
)

# --------------------------------------------------
# Load model files
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "bank_subscription_model_compressed_5.joblib"
    )

    scaler = joblib.load("scaler.pkl")

    columns = joblib.load("columns.pkl")

    return model, scaler, columns


model, scaler, columns = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title(" Bank Subscription Predictor")

st.write(
    "Enter the customer's information below "
    "to predict whether the customer will subscribe."
)


# --------------------------------------------------
# Customer information
# --------------------------------------------------

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
    [
        "married",
        "single",
        "divorced"
    ]
)

education = st.selectbox(
    "Education",
    [
        "primary",
        "secondary",
        "tertiary",
        "unknown"
    ]
)

default = st.selectbox(
    "Default",
    [
        "no",
        "yes"
    ]
)

balance = st.number_input(
    "Balance",
    value=0.0
)

housing = st.selectbox(
    "Housing Loan",
    [
        "no",
        "yes"
    ]
)

loan = st.selectbox(
    "Personal Loan",
    [
        "no",
        "yes"
    ]
)

contact = st.selectbox(
    "Contact",
    [
        "cellular",
        "telephone",
        "unknown"
    ]
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
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec"
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
    [
        "failure",
        "success",
        "unknown",
        "other"
    ]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict", type="primary"):

    # Create DataFrame
    input_data = {
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

    df_input = pd.DataFrame([input_data])

    # One-hot encoding
    df_input = pd.get_dummies(
        df_input,
        drop_first=True
    )

    # Match training columns
    df_input = df_input.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale
    scaled_input = scaler.transform(df_input)

    # Prediction
    prediction = model.predict(
        scaled_input
    )[0]

    # Probability
    probability = model.predict_proba(
        scaled_input
    )[0][1]

    # Display result

    if prediction == 1:

        st.success(
            " Customer is likely to subscribe!"
        )

    else:

        st.error(
            " Customer is unlikely to subscribe."
        )

    st.metric(
        "Probability of Subscription",
        f"{probability * 100:.2f}%"
    )