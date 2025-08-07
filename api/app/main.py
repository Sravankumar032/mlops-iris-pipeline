from fastapi import FastAPI
from api.app.predict import load_model, make_prediction
from api.app.schema import IrisRequest

app = FastAPI(title="Iris Prediction API")

model = load_model("models/LogisticRegression.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Predictor API!"}

@app.post("/predict")
def predict_species(input_data: IrisRequest):
    data = input_data.dict()
    prediction = make_prediction(model, data)
    return {"prediction": prediction}
