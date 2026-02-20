import streamlit as st
import pandas as pd
import joblib

st.title("Titanic Survival Predictor 🚢")

model = joblib.load("models/best_model.pkl")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Embarked", ["S", "Q", "C"])

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

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("Passenger Survived 🎉")
    else:
        st.error("Passenger Did Not Survive ❌")