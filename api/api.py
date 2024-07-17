from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
import uvicorn

# Description and model paths
Description = """
### Data Features

Input features:
- **PRG**: Plasma glucose level (mu U/ml)
- **PL**: Blood Work Result-1 (mu U/ml)
- **PR**: Blood Pressure (mm Hg)
- **SK**: Blood Work Result-2 (mm)
- **TS**: Blood Work Result-3 (mu U/ml)
- **M11**: Body mass index (weight in kg / (height in m)^2)
- **BD2**: Blood Work Result-4 (mu U/ml)
- **Age**: Patient's age (years)
- **Insurance**: Presence of insurance card (Yes/No)

Output feature:
- **Predictions**: **[0]** Negative, **[1]** Positive
- **Sepsis**: **Positive**: if a patient in ICU will develop sepsis, and **Negative**: otherwise
"""

# Load Models
forest = joblib.load("../toolkit/forest.joblib")
xgb = joblib.load("../toolkit/xgb.joblib")
label_encoder = joblib.load("../toolkit/encoder.joblib")

# Configure Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Building app
app = FastAPI(
    title="Sepsis Prediction API",
    description=Description
)

# Define the input data model
class SepsisFeatures(BaseModel):
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: float
    Insurance: str

# Root endpoint to check API status
@app.get('/')
def status_check():
    return {"Welcome to the sepsis prediction API"}

# Endpoint for predicting sepsis using Random Forest model
@app.post("/random_forest_prediction")
def predict_sepsis(data: SepsisFeatures):
    try:
        # Create a DataFrame from the input data
        df = pd.DataFrame([data.model_dump()])

        # Make prediction using the loaded forest model
        forest_prediction = forest.predict(df)[0]
        
        # Inverse transform the prediction using label encoder
        sepsis_prediction = label_encoder.inverse_transform([forest_prediction])[0]
        
        return {"Sepsis Prediction": sepsis_prediction}

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Endpoint for predicting sepsis using XGBoost model
@app.post("/xgboost_prediction")
def predict_sepsis_xgb(data: SepsisFeatures):
    try:
        df = pd.DataFrame([data.model_dump()])
        xgb_prediction = xgb.predict(df)[0]
        sepsis_prediction = label_encoder.inverse_transform([xgb_prediction])[0]
        return {"Sepsis Prediction": sepsis_prediction}

    except Exception as e:
        logger.error(f"XGBoost prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="XGBoost prediction failed")
    
if __name__ == "__main__":
    uvicorn.run("main:app",reload=True)
