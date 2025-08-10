import os
import logging
from fastapi import FastAPI
from api.app.predict import load_model, make_prediction
from api.app.schema import IrisRequest
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Iris Prediction API")

# Load model once when the app starts
model = load_model("models/LogisticRegression.pkl")

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "iris_predictions_total",
    "Total number of predictions made by the Iris Prediction API"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Predictor API!"}

@app.post("/predict")
def predict_species(input_data: IrisRequest):
    data = input_data.dict()
    prediction = make_prediction(model, data)

    # Increment Prometheus counter
    PREDICTION_COUNTER.inc()

    # Log the request and prediction
    logging.info(f"Request: {data} | Prediction: {prediction}")

    return {"prediction": prediction}

@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)