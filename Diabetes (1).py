import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random

# -----------------------------
# App Title
# -----------------------------
st.title("🩺 Diabetes Prediction Using Machine Learning ")

st.write("""
This app predicts **whether a patient is diabetic** based on health parameters.  
The model is trained on the **Pima Indians Diabetes Dataset**.

 Diabetes Prediction using Machine Learning "Diabetes" prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment.
""")
st.image('https://www.clinicaladvisor.com/wp-content/uploads/sites/11/2020/06/diabetes-care_G_1213259073.jpg')

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")  # Upload this dataset to your repo
    return df

df = load_data()

# -----------------------------
# Train Model
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.write(f"Model Accuracy: **{accuracy:.2f}**")

# -----------------------------
# Sidebar for User Input
# -----------------------------
st.sidebar.header("Enter Patient Data")
st.sidebar.image('https://aptivamedical.com/wp-content/uploads/2023/06/Detect-glucose-trends-and-patterns.jpg')

def user_input():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 80)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 10, 100, 33)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Patient Data")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

import time
random.seed(132)
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Diabetes') 

place = st.empty()
place.image('https://media1.tenor.com/m/wumcpBfLF4AAAAAC/heartbeat.gif',width = 200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)


st.subheader("Prediction Result")
st.write("🟥 Diabetic" if prediction[0] == 1 else "🟩 Not Diabetic")

st.markdown('Designed by:**Ashish Luthra**  and  **Mishti Sehgal**')









