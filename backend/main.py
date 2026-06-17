import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Titanic Survival Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)

try:
    logistic_model = joblib.load(os.path.join(BASE_DIR, "models", "logistic_model.pkl"))
    random_forest_model = joblib.load(os.path.join(BASE_DIR, "models", "random_forest_model.pkl"))
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    raise


class PassengerInput(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str
    model: str = "logistic"

    @validator("pclass")
    def valid_pclass(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("Passenger class must be 1, 2, or 3")
        return v

    @validator("sex")
    def valid_sex(cls, v):
        if v.lower() not in ["male", "female"]:
            raise ValueError("Sex must be male or female")
        return v.lower()

    @validator("embarked")
    def valid_embarked(cls, v):
        if v.upper() not in ["C", "Q", "S"]:
            raise ValueError("Embarked must be C, Q, or S")
        return v.upper()

    @validator("model")
    def valid_model(cls, v):
        if v not in ["logistic", "random_forest"]:
            raise ValueError("Model must be logistic or random_forest")
        return v


@app.post("/predict")
def predict(passenger: PassengerInput):
    selected_model = logistic_model if passenger.model == "logistic" else random_forest_model

    input_data = pd.DataFrame([{
        "Pclass": passenger.pclass,
        "Sex": 0 if passenger.sex == "male" else 1,
        "Age": passenger.age,
        "SibSp": passenger.sibsp,
        "Parch": passenger.parch,
        "Fare": passenger.fare,
        "Embarked_Q": 1 if passenger.embarked == "Q" else 0,
        "Embarked_S": 1 if passenger.embarked == "S" else 0,
    }])

    prediction = int(selected_model.predict(input_data)[0])
    probability = float(selected_model.predict_proba(input_data)[0][1])

    feature_importance = None
    if passenger.model == "random_forest":
        feature_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]
        importances = selected_model.feature_importances_.tolist()
        feature_importance = [
            {"feature": f, "importance": round(i, 4)}
            for f, i in zip(feature_names, importances)
        ]
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    logger.info(f"Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'} | Prob: {probability:.2f}")

    return {
        "survived": prediction == 1,
        "probability": round(probability * 100, 2),
        "model_used": passenger.model,
        "feature_importance": feature_importance,
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}