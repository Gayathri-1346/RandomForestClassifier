# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================================================
# LIGHT BACKGROUND UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f8ff;
}

.main-title {
    text-align: center;
    color: #1d3557;
    font-size: 42px;
    font-weight: bold;
}

.sub-text {
    text-align: center;
    color: #5c677d;
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
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<p class="main-title">🏦 Loan Approval Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-text">Predict Whether Loan Will Be Approved or Rejected</p>',
    unsafe_allow_html=True
)

# =========================================================
# CREATE MODEL FOLDER
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(BASE_DIR, "model")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

MODEL_PATH = os.path.join(model_dir, "loan_model.pkl")
ENCODER_PATH = os.path.join(model_dir, "label_encoders.pkl")

# =========================================================
# LOAD DATASET
# =========================================================

csv_path = os.path.join(BASE_DIR, "data", "Loan_Prediction.csv")

df = pd.read_csv(csv_path)

# =========================================================
# DATA CLEANING
# =========================================================

# Remove Loan_ID
if "Loan_ID" in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Remove spaces
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Replace empty strings
df.replace("", np.nan, inplace=True)

# Fill missing categorical values
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill missing numerical values
numerical_cols = df.select_dtypes(exclude="object").columns

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

# =========================================================
# LABEL ENCODING
# =========================================================

label_encoders = {}

for col in categorical_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X = X.astype(float)
y = y.astype(int)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# SAVE MODEL
# =========================================================

pickle.dump(model, open(MODEL_PATH, "wb"))
pickle.dump(label_encoders, open(ENCODER_PATH, "wb"))

# =========================================================
# ACCURACY
# =========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# =========================================================
# GRAPHS
# =========================================================

st.header("📊 Dataset Visualizations")

col1, col2 = st.columns(2)

# ---------------- LOAN STATUS GRAPH ----------------

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

# ---------------- APPLICANT INCOME GRAPH ----------------

with col2:

    st.subheader("Applicant Income Distribution")

    fig2, ax2 = plt.subplots()

    ax2.hist(df["ApplicantIncome"], bins=20)

    ax2.set_xlabel("Applicant Income")
    ax2.set_ylabel("Frequency")

    st.pyplot(fig2)

# =========================================================
# MORE GRAPHS
# =========================================================

col3, col4 = st.columns(2)

# ---------------- PROPERTY AREA ----------------

with col3:

    st.subheader("Property Area Distribution")

    fig3, ax3 = plt.subplots()

    df["Property_Area"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax3
    )

    ax3.set_ylabel("")

    st.pyplot(fig3)

# ---------------- CREDIT HISTORY ----------------

with col4:

    st.subheader("Credit History")

    fig4, ax4 = plt.subplots()

    df["Credit_History"].value_counts().plot(
        kind="bar",
        ax=ax4
    )

    ax4.set_xlabel("Credit History")
    ax4.set_ylabel("Count")

    st.pyplot(fig4)

# =========================================================
# USER INPUT SECTION
# =========================================================

st.header("📝 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    Married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    Education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    Self_Employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    ApplicantIncome = st.number_input(
        "Applicant Income",
        min_value=0
    )

with col2:

    CoapplicantIncome = st.number_input(
        "Coapplicant Income",
        min_value=0
    )

    LoanAmount = st.number_input(
        "Loan Amount",
        min_value=0
    )

    Loan_Amount_Term = st.number_input(
        "Loan Amount Term",
        min_value=0
    )

    Credit_History = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

    Property_Area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({
    "Gender": [Gender],
    "Married": [Married],
    "Dependents": [Dependents],
    "Education": [Education],
    "Self_Employed": [Self_Employed],
    "ApplicantIncome": [ApplicantIncome],
    "CoapplicantIncome": [CoapplicantIncome],
    "LoanAmount": [LoanAmount],
    "Loan_Amount_Term": [Loan_Amount_Term],
    "Credit_History": [Credit_History],
    "Property_Area": [Property_Area]
})

# =========================================================
# ENCODE INPUT
# =========================================================

for col in input_data.columns:

    if col in label_encoders:

        input_data[col] = label_encoders[col].transform(
            input_data[col]
        )

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("Predict Loan Status"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:

        st.markdown(
            """
            <div class="result-box"
            style="background-color:#d4edda;
            color:#155724;">
            ✅ Loan Approved
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-box"
            style="background-color:#f8d7da;
            color:#721c24;">
            ❌ Loan Rejected
            </div>
            """,
            unsafe_allow_html=True
        )
