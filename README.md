# Bank Customer Subscription Prediction

**End-to-End Machine Learning Application | Python • Scikit-learn • FastAPI • Streamlit**

An end-to-end machine learning application that predicts whether a bank customer is likely to subscribe to a banking product or service.

The project covers the complete workflow from **machine learning model development to API-based model serving and an interactive user interface**.

---

##  Overview

This project uses customer and marketing campaign information to predict the likelihood of a customer subscribing to a bank product.

The application provides:

* Machine learning classification
* Handling of imbalanced data using **SMOTE**
* Model comparison and evaluation
* Hyperparameter tuning
* Random Forest-based prediction
* Model serialization with Joblib
* REST API deployment using **FastAPI**
* Interactive frontend using **Streamlit**
* Prediction probability alongside the final prediction

### Example Output

```text
Prediction: YES
Probability of subscription: 65.43%
```

---

# Business Problem

Banks conduct marketing campaigns to promote financial products and services.

Contacting every customer equally can be inefficient. A predictive model can help identify customers who are more likely to subscribe, allowing marketing teams to prioritize their efforts.

The objective of this project is therefore to answer:

> **Is this customer likely to subscribe to the bank's product?**

The model produces both:

```text
Prediction → YES / NO
Probability → Probability of subscription
```

---

# Project Architecture

```text
                    ┌──────────────────────┐
                    │   Bank Customer Data │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    │ & Feature Engineering│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Training       │
                    │ & Evaluation         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ SMOTE +              │
                    │ Hyperparameter       │
                    │ Tuning                │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Random Forest Model  │
                    └──────────┬───────────┘
                               │
                         Joblib Serialization
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
      ┌───────────────┐                 ┌───────────────┐
      │    FastAPI    │                 │   Streamlit   │
      │   REST API    │◄────────────────│   Frontend    │
      └───────┬───────┘                 └───────────────┘
              │
              ▼
      ┌────────────────┐
      │ Prediction +   │
      │ Probability    │
      └────────────────┘
```

---

#  Machine Learning Workflow

## 1. Data Preparation

The dataset is cleaned and prepared for machine learning.

Categorical variables are converted into numerical features using one-hot encoding.

```python
pd.get_dummies()
```

---

## 2. Feature Selection

The `duration` feature is excluded from the prediction workflow because it represents the duration of the marketing call and is only known after the interaction.

Removing it helps avoid using information that would not be available when making a genuine pre-call prediction.

---

## 3. Train/Test Split

The dataset is divided into training and testing sets so that model performance can be evaluated on unseen data.

---

## 4. Feature Scaling

Numerical features are standardized using:

```python
StandardScaler()
```

The fitted scaler is saved for use when new customer data is submitted through the API.

---

## 5. Handling Class Imbalance

The project uses:

```text
SMOTE
```

(**Synthetic Minority Over-sampling Technique**) to address class imbalance in the training data.

---

#  Models

Multiple classification algorithms were explored:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

The final application uses the trained **Random Forest** model for prediction.

Hyperparameter optimization is performed using:

```text
GridSearchCV
```

---

#  Saved Model Components

The trained application uses three serialized files:

```text
bank_subscription_model.pkl
scaler.pkl
columns.pkl
```

### `bank_subscription_model.pkl`

Contains the trained machine learning model.

### `scaler.pkl`

Contains the fitted feature scaler used during model development.

### `columns.pkl`

Stores the feature columns generated during preprocessing so that incoming API data can be aligned with the same feature structure used during training.

---

#  FastAPI Backend

FastAPI is used to expose the machine learning model through a REST API.

### Endpoint

```http
POST /predict
```

The API accepts customer information and returns a prediction and probability.

### Example Request

```json
{
  "age": 35,
  "job": "technician",
  "marital": "married",
  "education": "secondary",
  "default": "no",
  "balance": 1200,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "aug",
  "campaign": 1,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}
```

### Example Response

```json
{
  "prediction": "yes",
  "probability_of_yes": 0.6543
}
```

---

#  Streamlit Frontend

The project also includes an interactive Streamlit interface.

Instead of manually sending JSON requests to the API, a user can enter customer information through a web interface.

```text
Customer Information
        ↓
     Predict
        ↓
    FastAPI API
        ↓
 Random Forest Model
        ↓
Prediction + Probability
```

This makes the machine learning model easier for a non-technical user to interact with.

---

#  Technology Stack

| Category              | Technologies                                                                            |
| --------------------- | --------------------------------------------------------------------------------------- |
| Language              | Python                                                                                  |
| Data Processing       | Pandas                                                                                  |
| Machine Learning      | Scikit-learn                                                                            |
| Imbalanced Learning   | imbalanced-learn / SMOTE                                                                |
| Models                | Random Forest, Logistic Regression, Decision Tree, Gradient Boosting, XGBoost, LightGBM |
| Hyperparameter Tuning | GridSearchCV                                                                            |
| Model Serialization   | Joblib                                                                                  |
| Backend               | FastAPI                                                                                 |
| Server                | Uvicorn                                                                                 |
| Frontend              | Streamlit                                                                               |
| API Testing           | FastAPI Swagger UI                                                                      |

---

#  Project Structure

```text
bank-subscription-prediction/
│
├── AdvanceML.ipynb
│
├── app.py
│
├── frontend.py
│
├── bank_subscription_model.pkl
│
├── scaler.pkl
│
├── columns.pkl
│
├── requirements.txt
│
└── README.md
```

---

#  Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/bank-subscription-prediction.git
```

Move into the project directory:

```bash
cd bank-subscription-prediction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

#  Run the Application

The project has two components: **FastAPI backend** and **Streamlit frontend**.

## 1. Start FastAPI

Open a terminal and run:

```bash
python -m uvicorn app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 2. Start Streamlit

Open a second terminal:

```bash
python -m streamlit run frontend.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🔌 API Workflow

When a user clicks **Predict**:

```text
Streamlit
    │
    │ Customer Data
    ▼
FastAPI /predict
    │
    ▼
DataFrame
    │
    ▼
One-Hot Encoding
    │
    ▼
Feature Alignment
    │
    ▼
StandardScaler
    │
    ▼
Random Forest
    │
    ▼
Prediction
    │
    ├── YES / NO
    │
    └── Probability
```

---

#  Example Prediction

For a sample customer, the application may return:

```json
{
  "prediction": "yes",
  "probability_of_yes": 0.6543
}
```

Interpretation:

> The model predicts that the customer belongs to the **YES** class, with an estimated subscription probability of **65.43%**.

The probability is a model estimate, not a guarantee of customer behavior.

---

#  Potential Business Application

A system like this could support bank marketing teams by helping them prioritize customers for campaigns.

For example:

```text
Customer A → 82% → Higher predicted likelihood
Customer B → 65% → Moderate predicted likelihood
Customer C → 18% → Lower predicted likelihood
```

The predictions could be incorporated into a broader customer segmentation or campaign-prioritization workflow.

---

#  Production Considerations

This repository demonstrates a **local end-to-end ML application** and is not presented as a production banking system.

A production implementation would require additional components such as:

* Authentication and authorization
* Secure API deployment
* Input validation
* Database integration
* Logging
* Monitoring
* Model versioning
* Model drift detection
* Data privacy controls
* Security controls
* CI/CD
* Cloud infrastructure
* Fairness and bias evaluation

---

#  Future Improvements

Planned improvements could include:

*  Cloud deployment
*  API authentication
*  Database integration
*  Prediction history
*  Model monitoring dashboard
*  Automated testing
*  CI/CD pipeline
*  Model versioning
*  Model drift monitoring
*  Improved Streamlit interface
*  More detailed prediction analytics

---

#  Skills Demonstrated

This project demonstrates practical experience in:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Classification
* Imbalanced learning
* SMOTE
* Model comparison
* Hyperparameter tuning
* Random Forest
* Model serialization
* REST API development
* FastAPI
* Streamlit
* Machine Learning model serving
* End-to-end ML application development

---

#  Author

**Uroosa Khan**

---

##  Project Summary

This project demonstrates the transition from a traditional machine learning notebook to an **interactive end-to-end ML application**:

```text
Raw Data
   ↓
Preprocessing
   ↓
Model Training
   ↓
SMOTE
   ↓
Hyperparameter Tuning
   ↓
Random Forest
   ↓
Model Serialization
   ↓
FastAPI REST API
   ↓
Streamlit Frontend
   ↓
Real-Time Prediction
```

**Built with Python, Scikit-learn, FastAPI, and Streamlit.**
