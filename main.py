from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os 
import joblib

# Load trained model
model = joblib.load("model.pkl")

# API Key
load_dotenv()
API_KEY = os.getenv("API_KEY")
print(f"Loaded API Key: {API_KEY}")

# Create FastAPI app
app = FastAPI(
    title="My ML Model API",
    version="1.0"
)

# Input schema
class InputData(BaseModel):
    Sepal_Length: float
    Sepal_Width: float
    Petal_Length: float
    Petal_Width: float

# Home route
@app.get("/")
def home():
    return {
        "message": "Model API is running"
    }

# Prediction route
@app.post("/predict")
def predict(
    data: InputData,
    x_api_key: str = Header(...)
):

    # Verify API key
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

    # Prepare input
    features = [[
        data.Sepal_Length,
        data.Sepal_Width,
        data.Petal_Length,
        data.Petal_Width
    ]]

    # Make prediction
    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }