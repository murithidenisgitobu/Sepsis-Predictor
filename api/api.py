from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os, uvicorn

app = FastAPI()

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
    Insurance: int

@app.get('/')
def status_check():
    return {"Welcome to the sepsis prediction API"}
