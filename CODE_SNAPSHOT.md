# Final Project Code Snapshot


## backend/main.py

```
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

```


## backend/requirements.txt

```
fastapi
uvicorn
joblib
pandas
numpy
scikit-learn
shap
mlflow
pydantic
imbalanced-learn

```


## backend/Dockerfile

```
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

```


## frontend/src/App.js

```
import React from "react";
import Home from "./pages/Home";

function App() {
  return <Home />;
}

export default App;

```


## frontend/src/index.js

```
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

```


## frontend/src/pages/Home.js

```
import React, { useState, useEffect } from "react";
import axios from "axios";

const API_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

const defaultForm = {
  Country: "USA",
  Year: 2022,
  Target_Industry: "Finance",
  Financial_Loss: 50.0,
  Number_of_Affected_Users: 100000,
  Attack_Source: "Nation-state",
  Security_Vulnerability_Type: "Unpatched Software",
  Defense_Mechanism_Used: "Firewall",
  Incident_Resolution_Time: 48,
};

export default function Home() {
  const [form, setForm]       = useState(defaultForm);
  const [options, setOptions] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    axios.get(`${API_URL}/options`)
      .then(res => setOptions(res.data))
      .catch(() => setError("Cannot connect to backend."));
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    const nums = ["Year","Financial_Loss","Number_of_Affected_Users","Incident_Resolution_Time"];
    setForm(p => ({ ...p, [name]: nums.includes(name) ? parseFloat(value) : value }));
  };

  const handleSubmit = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await axios.post(`${API_URL}/predict`, form);
      setResult(res.data);
    } catch { setError("Prediction failed. Check backend."); }
    setLoading(false);
  };

  const labelMap = {
    Country: "Country",
    Target_Industry: "Target Industry",
    Attack_Source: "Attack Source",
    Security_Vulnerability_Type: "Security Vulnerability Type",
    Defense_Mechanism_Used: "Defense Mechanism Used",
  };

  const isAttack = result?.verdict === "Attack";

  return (
    <div style={{ fontFamily:"Arial,sans-serif", background:"#f0f2f5", minHeight:"100vh" }}>
      <header style={{ background:"#1a1a2e", color:"white", padding:"20px 40px" }}>
        <h1 style={{ margin:0, fontSize:22 }}>🔒 Cybersecurity Attack Detector</h1>
        <p style={{ margin:"4px 0 0", color:"#aab", fontSize:13 }}>
          MOD10 Machine Learning · Binary Classification · Attack vs. Normal
        </p>
      </header>

      <div style={{ maxWidth:920, margin:"28px auto", padding:"0 20px" }}>
        {error && <div style={{ background:"#fee", border:"1px solid #e88", borderRadius:8, padding:"10px 16px", marginBottom:18, color:"#c00" }}>⚠️ {error}</div>}

        {/* Form */}
        <div style={{ background:"white", borderRadius:12, padding:26, boxShadow:"0 2px 12px rgba(0,0,0,0.08)", marginBottom:22 }}>
          <h2 style={{ marginTop:0, color:"#1a1a2e" }}>Enter Event Details</h2>
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:"14px 22px" }}>
            {Object.entries(labelMap).map(([field, label]) => (
              <div key={field}>
                <label style={{ display:"block", marginBottom:4, fontWeight:"bold", fontSize:13, color:"#444" }}>{label}</label>
                <select name={field} value={form[field]} onChange={handleChange}
                  style={{ width:"100%", padding:"8px 10px", borderRadius:6, border:"1px solid #ccc", fontSize:14 }}>
                  {options && options[label]
                    ? options[label].map(o => <option key={o}>{o}</option>)
                    : <option>{form[field]}</option>}
                </select>
              </div>
            ))}
            {[
              { name:"Year", label:"Year", min:2015, max:2024 },
              { name:"Financial_Loss", label:"Financial Loss (Million $)", min:0 },
              { name:"Number_of_Affected_Users", label:"Affected Users", min:0 },
              { name:"Incident_Resolution_Time", label:"Resolution Time (Hours)", min:0 },
            ].map(({ name, label, min, max }) => (
              <div key={name}>
                <label style={{ display:"block", marginBottom:4, fontWeight:"bold", fontSize:13, color:"#444" }}>{label}</label>
                <input type="number" name={name} value={form[name]} min={min} max={max}
                  onChange={handleChange}
                  style={{ width:"100%", padding:"8px 10px", borderRadius:6, border:"1px solid #ccc", fontSize:14, boxSizing:"border-box" }}/>
              </div>
            ))}
          </div>
          <button onClick={handleSubmit} disabled={loading}
            style={{ marginTop:22, padding:"11px 30px", fontSize:15, fontWeight:"bold",
              background:loading?"#aaa":"#1a1a2e", color:"white", border:"none", borderRadius:8, cursor:loading?"not-allowed":"pointer" }}>
            {loading ? "Analysing…" : "🔍 Detect Attack"}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div style={{ background:"white", borderRadius:12, padding:26, boxShadow:"0 2px 12px rgba(0,0,0,0.08)" }}>
            <h2 style={{ marginTop:0, color:"#1a1a2e" }}>Detection Result</h2>

            {/* Verdict banner */}
            <div style={{
              textAlign:"center", padding:"18px", borderRadius:10, marginBottom:22,
              background:isAttack?"#ffeaea":"#eafaf1",
              border:`2px solid ${isAttack?"#e74c3c":"#2ecc71"}`
            }}>
              <div style={{ fontSize:36 }}>{isAttack ? "🚨" : "✅"}</div>
              <div style={{ fontSize:26, fontWeight:"bold", color:isAttack?"#c0392b":"#27ae60" }}>
                {result.verdict}
              </div>
              <div style={{ color:"#555", marginTop:4 }}>
                Confidence: <strong>{result.confidence}%</strong> (Random Forest)
              </div>
            </div>

            {/* Model votes */}
            <h3 style={{ color:"#333", marginBottom:10 }}>Model Votes</h3>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:12, marginBottom:22 }}>
              {Object.entries(result.predictions).map(([model, label]) => (
                <div key={model} style={{
                  textAlign:"center", padding:"14px 10px", borderRadius:10,
                  background:label==="Attack"?"#ffeaea":"#eafaf1",
                  border:`1px solid ${label==="Attack"?"#e74c3c":"#2ecc71"}`
                }}>
                  <div style={{ fontSize:11, color:"#888", marginBottom:4 }}>
                    {model.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase())}
                  </div>
                  <div style={{ fontSize:17, fontWeight:"bold", color:label==="Attack"?"#c0392b":"#27ae60" }}>
                    {label}
                  </div>
                </div>
              ))}
            </div>

            {/* Probability bars */}
            <h3 style={{ color:"#333", marginBottom:10 }}>Attack Probability (Random Forest)</h3>
            {Object.entries(result.probabilities).map(([cls, prob]) => (
              <div key={cls} style={{ marginBottom:10 }}>
                <div style={{ display:"flex", justifyContent:"space-between", fontSize:13, marginBottom:3 }}>
                  <span>{cls}</span><span>{(prob*100).toFixed(1)}%</span>
                </div>
                <div style={{ background:"#eee", borderRadius:4, height:14 }}>
                  <div style={{
                    width:`${prob*100}%`, height:"100%", borderRadius:4, transition:"width 0.4s",
                    background:cls==="Attack"?"#e74c3c":"#2ecc71"
                  }}/>
                </div>
              </div>
            ))}

            {/* SHAP explanation */}
            <h3 style={{ color:"#333", marginTop:20, marginBottom:10 }}>
              🔍 Top Features Driving This Prediction (SHAP)
            </h3>
            <p style={{ color:"#666", fontSize:13, marginBottom:12 }}>
              Positive values push toward <strong>Attack</strong>. Negative values push toward <strong>Normal</strong>.
            </p>
            {result.shap_top_features.map(({ feature, impact }) => {
              const pct = Math.min(Math.abs(impact) * 400, 100);
              return (
                <div key={feature} style={{ marginBottom:10 }}>
                  <div style={{ display:"flex", justifyContent:"space-between", fontSize:13, marginBottom:3 }}>
                    <span>{feature.replace(/_enc$/,"").replace(/_/g," ")}</span>
                    <span style={{ color:impact>0?"#e74c3c":"#2ecc71", fontWeight:"bold" }}>
                      {impact>0?"+":""}{impact.toFixed(4)}
                    </span>
                  </div>
                  <div style={{ background:"#eee", borderRadius:4, height:12 }}>
                    <div style={{
                      width:`${pct}%`, height:"100%", borderRadius:4,
                      background:impact>0?"#e74c3c":"#2ecc71"
                    }}/>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
      <footer style={{ textAlign:"center", padding:14, color:"#888", fontSize:12 }}>
        MOD10 Machine Learning Project · Winter 2026 · Instructor: Mohammed A. Shehab
      </footer>
    </div>
  );
}

```


## frontend/package.json

```
{
  "name": "cybersecurity-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.2.0",
    "@testing-library/user-event": "^13.5.0",
    "axios": "^1.7.9",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "web-vitals": "^2.1.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
```


## frontend/Dockerfile

```
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

```


## docker-compose.yml

```
services:
  backend:
    build: ./backend
    container_name: cybersecurity-backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/models:/app/models
    restart: unless-stopped

  frontend:
    build: ./frontend
    container_name: cybersecurity-frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

```


## setup.sh

```
#!/bin/bash
set -e

echo ""
echo "=============================================="
echo "  Cybersecurity ML Project — Final Setup"
echo "=============================================="
echo ""

echo "[1/4] Installing Python dependencies..."
pip install --quiet pandas numpy scikit-learn matplotlib seaborn shap mlflow joblib notebook ipykernel imbalanced-learn fastapi uvicorn pydantic
echo "      ✅ Python packages installed."
echo ""

echo "[2/4] Running notebook to train models..."
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=900 \
  --output cybersecurity_ml_project_executed.ipynb \
  cybersecurity_ml_project.ipynb
echo "      ✅ Notebook executed successfully."
echo ""

echo "[3/4] Verifying model files..."
REQUIRED=(
  "backend/models/random_forest.pkl"
  "backend/models/logistic_regression.pkl"
  "backend/models/adaboost.pkl"
  "backend/models/scaler.pkl"
  "backend/models/label_encoders.pkl"
  "backend/models/feature_cols.pkl"
  "backend/models/median_resolution.pkl"
  "backend/models/shap_explainer.pkl"
)
ALL_OK=true
for f in "${REQUIRED[@]}"; do
  if [ -f "$f" ]; then echo "      ✅ $f"; else echo "      ❌ MISSING: $f"; ALL_OK=false; fi
done
if [ "$ALL_OK" = false ]; then echo "ERROR: Missing model files."; exit 1; fi
echo ""

echo "[4/4] Building and starting Docker containers..."
docker-compose up --build -d
echo ""
echo "=============================================="
echo "  ✅ Project is running!"
echo "=============================================="
echo "  Frontend : http://localhost:3000"
echo "  Backend  : http://localhost:8000"
echo "  API Docs : http://localhost:8000/docs"
echo "  MLflow   : run 'mlflow ui' then open http://localhost:5000"
echo "  Stop     : docker-compose down"
echo ""

```
