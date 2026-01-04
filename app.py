import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -------------------------------
# Load trained model and scaler
# -------------------------------
model = joblib.load("liver_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Liver Disease Prediction App")
st.write("Provide the following patient details to predict liver disease.")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Patient Information")

Age = st.sidebar.slider("Age", 1, 100, 30)
Gender = st.sidebar.radio("Gender", ("Male", "Female"))
TB = st.sidebar.number_input("Total Bilirubin (TB)", 0.0, 100.0, 0.5)
DB = st.sidebar.number_input("Direct Bilirubin (DB)", 0.0, 50.0, 0.2)
Alkphos = st.sidebar.number_input("Alkaline Phosphotase", 0.0, 5000.0, 200.0)
Sgpt = st.sidebar.number_input("SGPT", 0.0, 1000.0, 20.0)
Sgot = st.sidebar.number_input("SGOT", 0.0, 1000.0, 20.0)
TP = st.sidebar.number_input("Total Protein (TP)", 0.0, 20.0, 6.5)
ALB = st.sidebar.number_input("Albumin (ALB)", 0.0, 10.0, 3.5)
AG_Ratio = st.sidebar.number_input("A/G Ratio", 0.0, 5.0, 1.2)

# Map Gender
Gender = 1 if Gender == "Male" else 0

# Collect input into array
input_data = np.array([[Age, Gender, TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, AG_Ratio]])
input_scaled = scaler.transform(input_data)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Prediction: Patient is Healthy")
    else:
        st.error("⚠️ Prediction: Patient may have Liver Disease")

    st.subheader("Prediction Probability")
    st.write(f"Healthy: {probability[1]*100:.2f}%")
    st.write(f"Liver Disease: {probability[0]*100:.2f}%")

# -------------------------------
# Optional: Show input data
# -------------------------------
with st.expander("See Patient Data"):
    st.dataframe(pd.DataFrame(input_data, columns=['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','A/G Ratio']))
