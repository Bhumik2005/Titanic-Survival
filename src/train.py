import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="both",
                    help="Choose model: logistic, random_forest, both")
args = parser.parse_args()

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("data/train.csv")
print("Initial Shape:", df.shape)

# ---------------------------
# Data Cleaning
# ---------------------------

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

# ---------------------------
# Separate Features and Target
# ---------------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining size:", X_train.shape)
print("Testing size:", X_test.shape)


if args.model in ["logistic", "both"]:
    print("\nRunning Logistic Regression...\n")
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("Logistic Accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))


if args.model in ["random_forest", "both"]:
    print("\nRunning Random Forest...\n")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_pred)

print("\n=== Logistic Regression ===")
print("Accuracy:", log_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, log_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_pred))

# ============================================================
# RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n=== Random Forest ===")
print("Accuracy:", rf_accuracy)

# ---------------------------
# Compare Models
# ---------------------------
if rf_accuracy > log_accuracy:
    print("\nRandom Forest performs better.")
else:
    print("\nLogistic Regression performs better.")

import joblib

# Save best model
best_model = rf_model if rf_accuracy > log_accuracy else log_model

joblib.dump(best_model, "models/best_model.pkl")
print("\nBest model saved to models/best_model.pkl")

# ---------------------------
# Feature Importance (Random Forest)
# ---------------------------
import pandas as pd

importances = rf_model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n=== Feature Importance (Random Forest) ===")
print(feature_importance_df)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()
