import joblib
import pandas as pd

# load trained model
model = joblib.load("models/best_model.pkl")

def predict_survival(pclass, sex, age, sibsp, parch, fare):

    data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare
    }])

    prediction = model.predict(data)

    return prediction[0]