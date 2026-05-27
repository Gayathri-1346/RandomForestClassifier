# app.py
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# ---------------- LIGHT BACKGROUND UI ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f5f9ff;
}

.main-title {
    text-align: center;
    color: #1e3a5f;
    font-size: 40px;
    font-weight: bold;
}

.sub-text {
    text-align: center;
    color: #4a5568;
    font-size: 18px;
    margin-bottom: 30px;
}

div.stButton > button {
    background-color: #4f8bf9;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
}

div.stButton > button:hover {
    background-color: #346bd1;
}

.result-box {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<p class="main-title">🏦 Loan Approval Prediction</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="sub-text">Predict whether a loan will be Approved or Rejected using Random Forest Classifier</p>',
    unsafe_allow_html=True
)

# ---------------- CREATE MODEL FOLDER ----------------
if not os.path.exists("model"):
    os.makedirs("model")

MODEL_PATH = "model/loan_model.pkl"
ENCODER_PATH = "model/label_encoders.pkl"

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/Loan_Prediction.csv")

# ---------------- DATA PREPROCESSING ----------------
df.dropna(inplace=True)

# Remove Loan_ID if exists
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

label_encoders = {}

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ---------------- TRAIN MODEL ----------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- SAVE MODEL ----------------
pickle.dump(model, open(MODEL_PATH, "wb"))
pickle.dump(label_encoders, open(ENCODER_PATH, "wb"))

# ---------------- MODEL ACCURACY ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Trained Successfully | Accuracy: {accuracy*100:.2f}%")

# ---------------- INPUT SECTION ----------------
st.header("Enter Applicant Details")

Gender = st.selectbox("Gender", ["Male", "Female"])

Married = st.selectbox("Married", ["Yes", "No"])

Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

Education = st.selectbox("Education", ["Graduate", "Not Graduate"])

Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

ApplicantIncome = st.number_input("Applicant Income", min_value=0)

CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)

LoanAmount = st.number_input("Loan Amount", min_value=0)

Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)

Credit_History = st.selectbox("Credit History", [1.0, 0.0])

Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---------------- ENCODE INPUT ----------------
input_data = pd.DataFrame({
    'Gender': [Gender],
    'Married': [Married],
    'Dependents': [Dependents],
    'Education': [Education],
    'Self_Employed': [Self_Employed],
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Property_Area': [Property_Area]
})

# Apply label encoding
for col in input_data.columns:
    if col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

# ---------------- PREDICTION BUTTON ----------------
if st.button("Predict Loan Status"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.markdown(
            """
            <div class="result-box" style="background-color:#d4edda; color:#155724;">
                ✅ Loan Approved
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="result-box" style="background-color:#f8d7da; color:#721c24;">
                ❌ Loan Rejected
            </div>
            """,
            unsafe_allow_html=True
        )
# --------------------- GRAPHS -----------------------------

st.header("📊 Dataset Visualizations")

col1, col2 = st.columns(2)

# -------- GRAPH 1 : Loan Status Count --------
with col1:
    st.subheader("Loan Approval Distribution")

    fig1, ax1 = plt.subplots()

    df["Loan_Status"].value_counts().plot(
        kind="bar",
        ax=ax1
    )

    ax1.set_xlabel("Loan Status")
    ax1.set_ylabel("Count")

    st.pyplot(fig1)

# -------- GRAPH 2 : Applicant Income --------
with col2:
    st.subheader("Applicant Income Distribution")

    fig2, ax2 = plt.subplots()

    ax2.hist(df["ApplicantIncome"], bins=20)

    ax2.set_xlabel("Applicant Income")
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2)