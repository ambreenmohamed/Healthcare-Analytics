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

# Submit button
if st.sidebar.button("Predict Risk"):
    # Prepare the input for prediction
    input_features = np.array([[age, bmi, blood_sugar, cholesterol, crp, renal_function, systolic, diastolic]])
    prediction = ensemble_model.predict(input_features)
    risk_status = "At Risk" if prediction[0] == 1 else "Not at Risk"

    # Display prediction result
    st.header(f"Prediction: {risk_status}")

    # Display personalized insights
    if risk_status == "At Risk":
        st.subheader("⚠️ Personalized Recommendations:")
        st.write("- Maintain a healthy diet and exercise regularly.")
        st.write("- Reduce intake of high-cholesterol and high-sugar foods.")
        st.write("- Monitor blood pressure and consult a doctor regularly.")
    else:
        st.subheader("✅ Great Job!")
        st.write("Keep up the healthy lifestyle!")

    # Visualize input data
    st.subheader("Input Data Visualization")
    fig, ax = plt.subplots()
    categories = ['Age', 'BMI', 'Blood Sugar', 'Cholesterol', 'CRP', 'Renal Function', 'Systolic', 'Diastolic']
    values = [age, bmi, blood_sugar, cholesterol, crp, renal_function, systolic, diastolic]
    
    # Modify bar colors based on values
    colors = ['red' if value > threshold else 'skyblue' for value, threshold in zip(values, [120, 25, 140, 200, 3, 80, 120, 80])]
    ax.bar(categories, values, color=colors)
    
    ax.set_ylabel('Values')
    ax.set_title('Health Metrics')

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
