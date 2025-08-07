from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import mlflow

def evaluate_classification(y_true, y_pred, average='macro'):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics

def log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

def print_classification_report(y_true, y_pred):
    print("Classification Report:\n", classification_report(y_true, y_pred))