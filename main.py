from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

model = joblib.load("model.pkl")

app = FastAPI()

class InputData(BaseModel):
    features: List[float]

@app.get("/status")
def status():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    x = np.array(data.features).reshape(1, -1)
    pred = model.predict(x).tolist()
    return {"prediction": pred}
