# Verification Results

## Backend API test

I tested the FastAPI app directly with `TestClient`.

- `GET /health`: passed
- `GET /options`: passed
- `POST /predict`: passed

Example `/predict` response:

```json
{
  "task": "Normal vs Attack",
  "predictions": {
    "random_forest": "Normal",
    "logistic_regression": "Normal",
    "adaboost": "Normal"
  },
  "probabilities": {
    "Normal": 0.7962,
    "Attack": 0.2038
  },
  "verdict": "Normal",
  "confidence": 79.6
}
```

## Frontend-backend link

The frontend now uses:

```js
const API_URL =
  process.env.REACT_APP_BACKEND_URL ||
  `${window.location.protocol}//${window.location.hostname}:8000`;
```

This works when opening:

- `http://localhost:3000` for the React frontend
- `http://localhost:8000` for the FastAPI backend

It also avoids the browser-side mistake of using `http://backend:8000`, because `backend` is only resolvable inside the Docker network, not inside the user's browser.

## Files changed

- `frontend/src/pages/Home.js`
  - Improved backend URL handling.
  - Improved numeric parsing.
  - Improved API error display.

- `backend/requirements.txt`
  - Pinned package versions to reduce deployment issues with saved pickle/joblib model artifacts.

- `backend/Dockerfile`
  - Updated to Python 3.13 to match the model artifact environment more safely.


## Highest-Grade Upgrade Verification

Additional improvements were added after the first correction:

- Added `advanced_model_tuning.py` with `GridSearchCV`.
- Added `StratifiedKFold` cross-validation for model tuning.
- Added SMOTE inside the cross-validation pipeline to avoid validation leakage.
- Patched the notebook preprocessing order so `StandardScaler` is fitted after train/test split.
- Added `outputs/hyperparameter_tuning_results.csv`.
- Added `outputs/model_selection_decision.md`.
- Added `HIGH_GRADE_REQUIREMENTS_CHECKLIST.md`.
- Updated `README.md` and `FINAL_REPORT.md` to explain the model-selection decision.

The final deployed model remains Random Forest + SMOTE because the executed notebook reports the strongest holdout balance for F1-score and ROC-AUC.
