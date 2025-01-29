import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the model
with open('ensemble_model2.pkl', 'rb') as file:
    ensemble_model = pickle.load(file)

# App title
st.title("Health Risk Prediction App")
st.write("Predict whether a person is at risk based on their health metrics.")

# Input fields
st.sidebar.header("User Input Features")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5, step=0.1)
blood_sugar = st.sidebar.number_input("Blood Sugar Levels (mg/dL)", min_value=50.0, max_value=400.0, value=90.0, step=1.0)
cholesterol = st.sidebar.number_input("Cholesterol Levels (mg/dL)", min_value=100.0, max_value=400.0, value=180.0, step=1.0)
crp = st.sidebar.number_input("C-reactive Protein (CRP) (mg/L)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
renal_function = st.sidebar.number_input("Renal Function (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
systolic = st.sidebar.number_input("Systolic (mmHg)", min_value=90, max_value=180, value=120, step=1)
diastolic = st.sidebar.number_input("Diastolic (mmHg)", min_value=60, max_value=120, value=80, step=1)

# Normal ranges
thresholds = {
    "Age": 120,
    "BMI": 25,
    "Blood Sugar": 140,
    "Cholesterol": 200,
    "CRP": 3,
    "Renal Function": 80,
    "Systolic": 120,
    "Diastolic": 80
}

# Submit button
if st.sidebar.button("Predict Risk"):
    # Prepare the input for prediction
    input_features = np.array([[age, bmi, blood_sugar, cholesterol, crp, renal_function, systolic, diastolic]])
    prediction = ensemble_model.predict(input_features)
    risk_status = "At Risk" if prediction[0] == 1 else "Not at Risk"

    # Display prediction result
    st.header(f"Prediction: {risk_status}")

    # Identify specific risk factors
    risk_factors = []

    if bmi > thresholds["BMI"]:
        risk_factors.append("BMI is above the healthy range. Consider a balanced diet and regular exercise.")

    if blood_sugar > thresholds["Blood Sugar"]:
        risk_factors.append("Blood sugar is high. Reduce sugar intake and monitor glucose levels.")

    if cholesterol > thresholds["Cholesterol"]:
        risk_factors.append("Cholesterol level is high. Eat fiber-rich foods and reduce saturated fats.")

    if crp > thresholds["CRP"]:
        risk_factors.append("CRP is elevated, indicating inflammation. Consider an anti-inflammatory diet.")

    if renal_function < thresholds["Renal Function"]:
        risk_factors.append("Renal function is low. Stay hydrated and monitor kidney health.")

    if systolic > thresholds["Systolic"] or diastolic > thresholds["Diastolic"]:
        risk_factors.append("Blood pressure is high. Reduce salt intake and practice stress management.")

    # Display personalized recommendations
    st.subheader("📌 Personalized Recommendations:")

    if risk_factors:
        for recommendation in risk_factors:
            st.write(f"- {recommendation}")
    else:
        st.write("✅ All your health metrics are within normal range. Keep up the good work!")


    # Visualize input data
    st.subheader("Input Data Visualization")
    fig, ax = plt.subplots()
    categories = list(thresholds.keys())
    values = [age, bmi, blood_sugar, cholesterol, crp, renal_function, systolic, diastolic]
    
    # Splitting normal and exceeding parts
    normal_values = [min(value, thresholds[cat]) for value, cat in zip(values, categories)]
    exceeding_values = [max(value - thresholds[cat], 0) for value, cat in zip(values, categories)]

    # Normal bars
    ax.bar(categories, normal_values, color="skyblue", label="Normal Range")

    # Exceeding stacked bars
    ax.bar(categories, exceeding_values, bottom=normal_values, color="red", label="Above Normal")

    ax.set_ylabel("Values")
    ax.set_title("Health Metrics")
    ax.legend()

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    st.pyplot(fig)

# Additional visuals or information
st.sidebar.header("Additional Information")
st.sidebar.write("### Normal Ranges for Reference:")
st.sidebar.write("- Blood Sugar: 70-140 mg/dL")
st.sidebar.write("- Cholesterol: <200 mg/dL")
st.sidebar.write("- CRP: <3 mg/L")
st.sidebar.write("- Systolic: 90-120 mmHg")
st.sidebar.write("- Diastolic: 60-80 mmHg")

st.sidebar.write("### About this App:")
st.sidebar.write("This tool predicts health risk based on common metrics. "
                 "The prediction is powered by an ensemble machine learning model.")

# Input validation
if blood_sugar < 70 or blood_sugar > 140:
    st.warning("⚠️ Blood Sugar level is outside the normal range (70-140 mg/dL).")
if cholesterol > 200:
    st.warning("⚠️ Cholesterol level is above the normal range (<200 mg/dL).")
if crp > 3:
    st.warning("⚠️ CRP level is above the normal range (<3 mg/L).")
if systolic < 90 or systolic > 120:
    st.warning("⚠️ Systolic blood pressure is outside the normal range (90-120 mmHg).")
if diastolic < 60 or diastolic > 80:
    st.warning("⚠️ Diastolic blood pressure is outside the normal range (60-80 mmHg).")
