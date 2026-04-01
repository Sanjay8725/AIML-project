from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "linear_regression_model.pkl"
MAX_PERCENT = 99.0


class PredictionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Adsorbent: str
    Metal: str
    dosage_g_l: float = Field(alias="Dosage (g/L)")
    temp_c: float = Field(alias="Temp (°C)")
    pH: float
    time_min: float = Field(alias="Time (min)")
    RPM: float
    c0_mg_l: float = Field(alias="C0 (mg/L)")


app = FastAPI(title="Heavy Metal Removal Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


try:
    model = load_model()
except Exception as exc:
    model = None
    model_load_error = str(exc)
else:
    model_load_error = None


@app.get("/health")
def health_check():
    if model is None:
        return {"status": "error", "model_loaded": False, "detail": model_load_error}
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(payload: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {model_load_error}")

    try:
        input_frame = pd.DataFrame(
            [
                {
                    "Adsorbent": payload.Adsorbent,
                    "Metal": payload.Metal,
                    "Dosage (g/L)": payload.dosage_g_l,
                    "Temp (°C)": payload.temp_c,
                    "pH": payload.pH,
                    "Time (min)": payload.time_min,
                    "RPM": payload.RPM,
                    "C0 (mg/L)": payload.c0_mg_l,
                }
            ]
        )

        raw_prediction = model.predict(input_frame)[0]
        clipped = float(np.round(np.clip(raw_prediction, 0, MAX_PERCENT), 1))

        return {
            "predicted_removal_percentage": clipped,
            "predicted_removal_percentage (%)": clipped,
            "model": "Linear Regression",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
