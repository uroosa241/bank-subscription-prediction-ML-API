from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load saved files
model = joblib.load("bank_subscription_model_compressed_5.joblib")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


# Customer information
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str


@app.post("/predict")
def predict(data: CustomerData):

    # Convert customer data into DataFrame
    df_input = pd.DataFrame([data.model_dump()])

    # One-hot encode categorical variables
    df_input = pd.get_dummies(df_input, drop_first=True)

    # Make columns exactly the same as training data
    df_input = df_input.reindex(columns=columns, fill_value=0)

    # Scale the data
    scaled_input = scaler.transform(df_input)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Probability
    probability = model.predict_proba(scaled_input)[0][1]

    return {
        "prediction": "yes" if prediction == 1 else "no",
        "probability_of_yes": round(float(probability), 4)
    }