import joblib
import numpy as np
import pandas as pd

def load_model(model_path: str):
    return joblib.load(model_path)

LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

def make_prediction(model, input_data):
    df = pd.DataFrame([input_data])
    df.columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
    prediction = model.predict(df)
    return LABELS[int(prediction[0])]  