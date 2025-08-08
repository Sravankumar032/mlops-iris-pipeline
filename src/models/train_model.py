import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.utils.metrics import (
    evaluate_classification,
    log_metrics,
    print_confusion_matrix,
    print_classification_report,
)

# =============================
# Path Setup for Codespaces
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/models
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))  # repo root

# Store MLflow runs in repo under artifacts/mlruns
MLFLOW_ARTIFACT_PATH = os.path.join(REPO_ROOT, "artifacts", "mlruns")
os.makedirs(MLFLOW_ARTIFACT_PATH, exist_ok=True)

# Store .pkl models in artifacts/models
MODEL_DIR = os.path.join(REPO_ROOT, "artifacts", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Ensure MLflow only uses this path
os.environ["MLFLOW_TRACKING_URI"] = f"file://{MLFLOW_ARTIFACT_PATH}"


def load_and_prepare_data(file_path: str, target_col: str):
    """Load CSV and encode target labels."""
    df = pd.read_csv(file_path)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Iris_Classification")

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train model, evaluate, save locally, and log with MLflow."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print and log metrics
    print_confusion_matrix(y_test, y_pred)
    print_classification_report(y_test, y_pred)
    metrics = evaluate_classification(y_test, y_pred)

    # Save locally in artifacts/models
    local_model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(model, local_model_path)
    print(f"Model saved at: {local_model_path}")

    # Log with MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(model.get_params())
        log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )


if __name__ == "__main__":
    # Adjust dataset path and target column as per your repo
    dataset_path = os.path.join(REPO_ROOT, "data", "processed", "processed.csv")
    target_column = "target"  

    X_train, X_test, y_train, y_test = load_and_prepare_data(dataset_path, target_column)

    # Train & log both models
    train_and_log_model(LogisticRegression(max_iter=200), "LogisticRegression", X_train, X_test, y_train, y_test)
    train_and_log_model(RandomForestClassifier(n_estimators=100), "RandomForestClassifier", X_train, X_test, y_train, y_test)