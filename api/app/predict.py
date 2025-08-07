import joblib
import numpy as np

def load_model(model_path: str):
    return joblib.load(model_path)

def make_prediction(model, data: dict):
    features = np.array([[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]])
    prediction = model.predict(features)[0]
    return prediction
