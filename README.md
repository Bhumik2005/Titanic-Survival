# Titanic Survival Prediction 🚢

## 1️⃣ Project Overview
This project builds an end-to-end Machine Learning pipeline to predict passenger survival using the Kaggle Titanic dataset.  
The workflow includes data cleaning, feature engineering, model training, evaluation, and feature importance analysis.

Models Used:
- Logistic Regression (Accuracy: ~81%)
- Random Forest (Accuracy: ~80%)

Key Insight:
Fare, Sex, and Age were the most important features influencing survival.

---

📌 Features Used

-Pclass

-Sex

-Age

-SibSp

-Parch

-Fare

-Embarked

🤖 Models Compared

-Logistic Regression

-Tuned Random Forest (GridSearchCV)

📊 Model Performance

-Best Accuracy: ~82%

-Cross Validation Accuracy: ~81.7%

-Evaluation Metrics:

  -Classification Report

  -Confusion Matrix

  -ROC Curve

  -Feature Importance

## 2️⃣ Tech Stack & Workflow
Technologies used:
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- joblib

📂 Project Structure

titanic-survival/
│
├── data/
│   └── train.csv
├── models/
│   └── best_model.pkl
├── src/
│   └── train.py
├── app.py
├── requirements.txt
└── README.md


Workflow:
1. Data preprocessing (handling missing values, encoding)
2. Train-test split
3. Model training & comparison
4. Performance evaluation (accuracy, confusion matrix)
5. Feature importance visualization
6. Best model saved automatically

---

## How to Run

1. Activate virtual environment:

```
..\ml_env\Scripts\activate
```


2. Run the training script:

```
python src/train.py
```


---

👨‍💻 Author: Bhumik Kumta  
GitHub: https://github.com/Bhumik2005  
LinkedIn: https://www.linkedin.com/in/bhumik-kumta-/




