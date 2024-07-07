from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder


app = FastAPI()


class sepsis_features(BaseModel):
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: float
    Insurance: int


@app.get('/')
def status_check():
    return {"Status": "Api is online...."}


forest = joblib.load('Models/tuned_rfc_model.joblib')
xgb = joblib.load('Models/tuned_xgb_model.joblib')
label_encoder = LabelEncoder()


@app.post('/forest_prediction')
def forest_prediction(data: sepsis_features):

    df = pd.DataFrame([data.model_dump()])

    prediction  = forest.predict(df)

    prediction = int(prediction[0])

    prediction = label_encoder.inverse_transform(prediction)[0]

    return {"Prediction": prediction}

    