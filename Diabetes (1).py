import streamlit as st
import pandas as pd
import pickle
import random
import time

# App title
st.header('🩺 Diabetes Prediction Using Machine Learning')

st.markdown("""
This app predicts whether a person has **Diabetes** or not based on health parameters.  
Choose a **machine learning model** and enter feature values to get predictions.
""")

st.image('https://cdn-icons-png.flaticon.com/512/616/616408.png', width=150)

# Sidebar - model selection
st.sidebar.header('⚙️ Settings')
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ("Logistic Regression", "Decision Tree", "Random Forest")
)

# Load selected model
if model_choice == "Logistic Regression":
    with open("logistic_regression.pkl", "rb") as f:
        model = pickle.load(f)
elif model_choice == "Decision Tree":
    with open("decision_tree.pkl", "rb") as f:
        model = pickle.load(f)
else:
    with open("random_forest.pkl", "rb") as f:
        model = pickle.load(f)

# Load Diabetes dataset (for feature ranges)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

st.sidebar.subheader("Set Feature Values")

all_values = []

# Create sliders for each feature
for col in df.columns[:-1]:  # exclude "Outcome"
    min_value, max_value = df[col].agg(['min', 'max'])
    default_value = random.randint(int(min_value), int(max_value))
    var = st.sidebar.slider(f"{col}", int(min_value), int(max_value), default_value)
    all_values.append(var)

final_value = [all_values]

# Predict
if st.sidebar.button("🔍 Predict"):
    ans = model.predict(final_value)[0]

    progress_bar = st.progress(0)
    placeholder = st.empty()
    placeholder.subheader(f'Predicting Diabetes using {model_choice}...') 

    place = st.empty()
    place.image('https://media1.tenor.com/m/9bgrt0D6BfIAAAAC/diabetes-blood-sugar.gif', width=200)

    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    placeholder.empty()
    place.empty()

    if ans == 0:
        st.success("✅ No Diabetes Detected")
    else:
        st.warning("⚠️ Diabetes Found")

st.markdown('Designed by: **Ashish Luthra**')






