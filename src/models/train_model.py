import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.utils.metrics import (
    evaluate_classification,
    log_metrics,
    print_confusion_matrix,
    print_classification_report,
)

# -----------------------
# Safe model directory inside project
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/models/
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))  # root of repo
MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "iris.csv"))

# Encode target if needed
if df["target"].dtype == "object":
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target"])

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Iris_Classification")

def train_and_log_model(model, model_name):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = evaluate_classification(y_test, preds)
        log_metrics(metrics)

        print(f"\nModel: {model_name}")
        print_confusion_matrix(y_test, preds)
        print_classification_report(y_test, preds)

        # Save model safely inside artifacts/models/
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved at: {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

if __name__ == "__main__":
    train_and_log_model(LogisticRegression(max_iter=200), "LogisticRegression")
    train_and_log_model(RandomForestClassifier(n_estimators=100), "RandomForest")