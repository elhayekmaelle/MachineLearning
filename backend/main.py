"""
Cybersecurity Attack Detection API
MOD10 – Machine Learning, Winter 2026
Instructor: Mohammed A. Shehab

Final backend version:
- Binary classification: Normal vs Attack
- Robust categorical encoding
- Feature reindexing to avoid feature-order mismatch
- SHAP explanation returned with each prediction
"""

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="Cybersecurity Attack Detection API",
    version="2.0.0",
    description="Predicts whether a cybersecurity event is Normal or Attack."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_artifact(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {path}. "
            "Run the notebook first to generate files in backend/models/."
        )
    return joblib.load(path)


try:
    rf_model = load_artifact("random_forest.pkl")
    lr_model = load_artifact("logistic_regression.pkl")
    ada_model = load_artifact("adaboost.pkl")
    scaler = load_artifact("scaler.pkl")
    le_dict = load_artifact("label_encoders.pkl")
    feature_cols = load_artifact("feature_cols.pkl")
    median_resolution = load_artifact("median_resolution.pkl")
    explainer = load_artifact("shap_explainer.pkl")
except Exception as exc:
    # Keep app importable, but report the error on endpoints.
    rf_model = lr_model = ada_model = scaler = le_dict = feature_cols = median_resolution = explainer = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None


class ThreatInput(BaseModel):
    Country: str = Field(..., example="USA")
    Year: int = Field(..., example=2022)
    Target_Industry: str = Field(..., example="Finance")
    Financial_Loss: float = Field(..., example=80.5, alias="Financial_Loss")
    Number_of_Affected_Users: int = Field(..., example=500000, alias="Number_of_Affected_Users")
    Attack_Source: str = Field(..., example="Nation-state")
    Security_Vulnerability_Type: str = Field(..., example="Unpatched Software")
    Defense_Mechanism_Used: str = Field(..., example="Firewall")
    Incident_Resolution_Time: int = Field(..., example=48)

    class Config:
        populate_by_name = True


def ensure_models_loaded():
    if MODEL_LOAD_ERROR is not None:
        raise HTTPException(status_code=500, detail=MODEL_LOAD_ERROR)


def safe_encode(label_encoder, value) -> int:
    """Safely encode categorical values.

    If the API receives an unseen category, map it to the first known class.
    This avoids crashing the API while keeping the feature vector valid.
    """
    value = str(value)
    if value in label_encoder.classes_:
        return int(label_encoder.transform([value])[0])
    fallback = label_encoder.classes_[0]
    return int(label_encoder.transform([fallback])[0])


def encode_input(data: ThreatInput):
    ensure_models_loaded()

    loss_per_user = data.Financial_Loss / (data.Number_of_Affected_Users + 1)
    high_resolution_time = int(data.Incident_Resolution_Time > median_resolution)

    row = {
        "Year": data.Year,
        "Financial Loss (in Million $)": data.Financial_Loss,
        "Number of Affected Users": data.Number_of_Affected_Users,
        "Incident Resolution Time (in Hours)": data.Incident_Resolution_Time,
        "Loss_per_User": loss_per_user,
        "High_Resolution_Time": high_resolution_time,
        "Country_enc": safe_encode(le_dict["Country"], data.Country),
        "Target Industry_enc": safe_encode(le_dict["Target Industry"], data.Target_Industry),
        "Attack Source_enc": safe_encode(le_dict["Attack Source"], data.Attack_Source),
        "Security Vulnerability Type_enc": safe_encode(
            le_dict["Security Vulnerability Type"],
            data.Security_Vulnerability_Type
        ),
        "Defense Mechanism Used_enc": safe_encode(
            le_dict["Defense Mechanism Used"],
            data.Defense_Mechanism_Used
        ),
    }

    X_raw = pd.DataFrame([row])

    # Critical safety step: enforce exactly the same columns/order used during training.
    X_raw = X_raw.reindex(columns=feature_cols, fill_value=0)

    X_scaled = scaler.transform(X_raw)
    return X_scaled, X_raw


@app.get("/")
def home():
    return {
        "message": "Cybersecurity Attack Detection API",
        "task": "Binary classification: Normal vs Attack",
        "docs": "/docs",
        "model_loaded": MODEL_LOAD_ERROR is None
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL_LOAD_ERROR is None else "model_artifacts_missing",
        "detail": MODEL_LOAD_ERROR
    }


@app.get("/options")
def get_options():
    ensure_models_loaded()
    return {col: list(le.classes_) for col, le in le_dict.items()}


@app.post("/predict")
def predict(data: ThreatInput):
    try:
        X_scaled, _ = encode_input(data)
        X_df = pd.DataFrame(X_scaled, columns=feature_cols)

        rf_pred = int(rf_model.predict(X_scaled)[0])
        lr_pred = int(lr_model.predict(X_scaled)[0])
        ada_pred = int(ada_model.predict(X_scaled)[0])

        rf_proba = rf_model.predict_proba(X_scaled)[0]

        label_map = {0: "Normal", 1: "Attack"}

        shap_values = explainer.shap_values(X_df)

        # SHAP output format changes depending on SHAP/sklearn versions.
        if isinstance(shap_values, list):
            sv_attack = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv_attack = shap_values[:, :, 1]
        else:
            sv_attack = shap_values

        sv_flat = np.ravel(sv_attack)
        shap_explanation = {
            feature_cols[i]: round(float(sv_flat[i]), 4)
            for i in range(min(len(feature_cols), len(sv_flat)))
        }

        top_features = sorted(
            shap_explanation.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:5]

        return {
            "task": "Normal vs Attack",
            "predictions": {
                "random_forest": label_map[rf_pred],
                "logistic_regression": label_map[lr_pred],
                "adaboost": label_map[ada_pred],
            },
            "probabilities": {
                "Normal": round(float(rf_proba[0]), 4),
                "Attack": round(float(rf_proba[1]), 4),
            },
            "verdict": label_map[rf_pred],
            "confidence": round(float(max(rf_proba)) * 100, 1),
            "shap_top_features": [
                {"feature": feature, "impact": impact}
                for feature, impact in top_features
            ],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
