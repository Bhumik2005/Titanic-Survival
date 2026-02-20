import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="centered"
)

# ---------------------------
# Header
# ---------------------------
st.title("Titanic Survival Prediction")
st.caption("Machine Learning Model for Passenger Survival Classification")

st.markdown("---")

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("models/best_model.pkl")

st.markdown("---")
st.subheader("Model Insights")

if hasattr(model, "feature_importances_"):

    feature_names = [
        "Pclass", "Sex", "Age", "SibSp",
        "Parch", "Fare", "Embarked_Q", "Embarked_S"
    ]

    importances = model.feature_importances_

    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# ---------------------------
# Input Section
# ---------------------------
st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)

with col2:
    sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents / Children", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)

embarked = st.selectbox("Embarked Port", ["S", "Q", "C"])

# ---------------------------
# Data Preparation
# ---------------------------
sex = 0 if sex == "Male" else 1
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked_Q": embarked_q,
    "Embarked_S": embarked_s
}])

st.markdown("---")

# ---------------------------
# Prediction Section
# ---------------------------
if st.button("Predict", use_container_width=True):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did Not Survive")

    st.metric("Survival Probability", f"{probability:.2%}")