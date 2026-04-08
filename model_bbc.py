from fastapi import FastAPI
import joblib
from pydantic import BaseModel


model_bbc = joblib.load("bbc_news_model.pkl")


app = FastAPI(title="BBC NEWS CLASSIFICATION PROJECT WITH FAST API")


class NewsInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "BBC NEWS PREDICTION PROJECT"}


@app.post("/predict")
def predict(data: NewsInput):
    prediction = model_bbc.predict([data.text])[0]
    
    return {
        "input_text": data.text,
        "predicted_category": prediction
    }