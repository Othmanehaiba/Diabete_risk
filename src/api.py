from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# 1. Initialisation de l'application FastAPI
app = FastAPI(
    title="Diabetes Risk Prediction API",
    description="API de prédiction du risque de diabète basée sur le modèle MLOps",
    version="1.0.0"
)

# 2. Chargement du pipeline champion
MODEL_PATH = "src/models/best_diabetes_pipeline.joblib"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/best_diabetes_pipeline.joblib"

pipeline = joblib.load(MODEL_PATH)

# 3. Schéma de validation des données d'entrée (Pydantic)
class PatientData(BaseModel):
    Pregnancies: int = Field(..., example=2)
    Glucose: float = Field(..., example=120.0)
    BloodPressure: float = Field(..., example=70.0)
    SkinThickness: float = Field(..., example=20.0)
    Insulin: float = Field(..., example=85.0)
    BMI: float = Field(..., example=28.5)
    DiabetesPedigreeFunction: float = Field(..., example=0.45)
    Age: int = Field(..., example=45)

# 4. Endpoints API
@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": pipeline is not None}

@app.post("/predict")
def predict_risk(data: PatientData):
    try:
        # Convertir les données entrantes en DataFrame Pandas
        input_data = pd.DataFrame([data.dict()])
        
        # Prédiction (Classe + Probabilité)
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0][1]
        
        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        
        return {
            "prediction": int(prediction),
            "risk_label": risk_label,
            "probability": float(round(probability, 4))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))