import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/train.csv")
print("Initial Shape:", df.shape)

# ============================================================
# DATA CLEANING
# ============================================================

# Drop Cabin (too many missing values)
df = df.drop(columns=["Cabin"])

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Convert Sex to numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# One-hot encode Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Drop unnecessary columns
df = df.drop(columns=["Name", "Ticket", "PassengerId"])

print("Cleaned Shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ============================================================
# FEATURES & TARGET
# ============================================================

X = df.drop("Survived", axis=1)
y = df["Survived"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)

# ============================================================
# HYPERPARAMETER TUNING (Random Forest)
# ============================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

rf_tuned = grid.best_estimator_

# ============================================================
# TRAIN MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest (Tuned)": rf_tuned
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc}")

    # Save individual models
    if name == "Logistic Regression":
        joblib.dump(model, "models/logistic_model.pkl")

    if "Random Forest" in name:
        joblib.dump(model, "models/random_forest_model.pkl")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print("\nBest Model Accuracy:", best_accuracy)

# Save best model
joblib.dump(best_model, "models/best_model.pkl")
print("All models saved successfully!")

# ============================================================
# CROSS VALIDATION
# ============================================================

scores = cross_val_score(best_model, X, y, cv=5)
print("Cross Validation Accuracy:", scores.mean())

# ============================================================
# EVALUATION
# ============================================================

y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Heatmap
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# ROC CURVE
# ============================================================

y_prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ============================================================
# SAMPLE PREDICTION FUNCTION
# ============================================================

def predict_new_passenger(data_dict):
    model = joblib.load("models/best_model.pkl")
    input_df = pd.DataFrame([data_dict])
    prediction = model.predict(input_df)
    return "Survived" if prediction[0] == 1 else "Did Not Survive"


sample_passenger = {
    "Pclass": 3,
    "Sex": 0,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked_Q": 0,
    "Embarked_S": 1
}

print("\nSample Passenger Prediction:")
print(predict_new_passenger(sample_passenger))