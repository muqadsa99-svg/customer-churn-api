from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Customer Churn API is Live"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"Churn_Risk": int(prediction)}
