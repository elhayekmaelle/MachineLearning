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
