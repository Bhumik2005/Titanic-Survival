import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title("🚢 Titanic Survival Prediction App")
st.write("Predict whether a passenger would survive the Titanic disaster.")

# --------------------------------------------------
# Sidebar - Model Selection
# --------------------------------------------------
st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"],
    key="model_selector"
)

# Load selected model
if model_choice == "Logistic Regression":
    model = joblib.load("models/logistic_model.pkl")
else:
    model = joblib.load("models/random_forest_model.pkl")

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.header("Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age", 1, 80, 25)

with col2:
    sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 500.0, 50.0)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("Predict Survival"):

    input_data = {
        "Pclass": pclass,
        "Sex": 0 if sex == "Male" else 1,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked_Q": 1 if embarked == "Q" else 0,
        "Embarked_S": 1 if embarked == "S" else 0
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Survived")
    else:
        st.error("❌ Did Not Survive")

    st.info(f"Survival Probability: {probability*100:.2f}%")

    # --------------------------------------------------
    # Feature Importance (Only for Random Forest)
    # --------------------------------------------------
    if model_choice == "Random Forest":
        st.subheader("Feature Importance")

        feature_names = input_df.columns
        importances = model.feature_importances_

        fig, ax = plt.subplots()
        ax.barh(feature_names, importances)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance - Random Forest")

        st.pyplot(fig)