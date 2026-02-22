import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Predict whether a passenger would survive the Titanic disaster.")
st.markdown("---")

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
logistic_model = joblib.load("models/logistic_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")

# ---------------------------------------------------
# Sidebar - Model Selection
# ---------------------------------------------------
st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("Logistic Regression", "Random Forest")
)

if model_choice == "Logistic Regression":
    model = logistic_model
else:
    model = rf_model

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
st.header("Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)

embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Convert inputs to model format
sex = 0 if sex == "Male" else 1
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_data = {
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked_Q": embarked_Q,
    "Embarked_S": embarked_S
}

input_df = pd.DataFrame([input_data])

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict Survival"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")

    st.info(f"🎯 Survival Probability: {probability:.2%}")

    # ---------------------------------------------------
    # Feature Importance (Only for Random Forest)
    # ---------------------------------------------------
    if model_choice == "Random Forest":

        st.markdown("---")
        st.subheader("Feature Importance")

        importances = model.feature_importances_
        features = input_df.columns

        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_title("Feature Importance")

        st.pyplot(fig)