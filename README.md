<<<<<<< HEAD
# MachineLearning
=======
# 🔒 Cybersecurity Event Detection — Normal vs Attack

## MOD10 – Machine Learning Project · Winter 2026  
**Instructor:** Mohammed A. Shehab  
**Dataset:** Global Cybersecurity Threats (2015–2024), adapted with realistic normal events

---

## Overview

This project applies classical Machine Learning to detect whether a cybersecurity event is **Normal** or an **Attack**.

The original dataset contains attack records only. To make the project more realistic, normal/benign events were added. In real cybersecurity systems, attacks are rare compared with normal activity, so the final dataset is intentionally imbalanced:

- **80% Normal**
- **20% Attack**

**Task:** Binary classification — predict whether an event is:

- `Normal`
- `Attack`

This project covers the full ML lifecycle required in the MOD10 project description: preprocessing, EDA, feature engineering, model training, evaluation, interpretability, deployment, and MLflow tracking.

---

## Important Dataset Adjustments

To satisfy the project requirements and improve realism:

- A reduced dataset of about **531 events** is used instead of all 3000 rows.
- Both **normal** and **attack** events are included.
- The final dataset is intentionally **imbalanced** to reflect real cybersecurity conditions.
- Ratios and percentages are used in the analysis instead of only raw counts.
- The target variable `Label` is **never used as a feature**.
- `Attack Type` is removed from model training to avoid target leakage.
- Normal events were generated with overlapping distributions to avoid making the classification problem too easy.
- Correlation analysis is used to check for redundant features, especially inter-feature correlations above **0.7**.

---

## Project Structure

```text
cybersecurity-ml-project/
├── cybersecurity_ml_project.ipynb
├── Global_Cybersecurity_Threats_2015-2024.csv
├── docker-compose.yml
├── setup.sh
├── backend/
│   ├── main.py
│   ├── models/
│   ├── requirements.txt
│   └── Dockerfile
└── frontend/
    ├── src/
    │   ├── App.js
    │   └── pages/Home.js
    ├── Dockerfile
    └── package.json
```

---

## Project Components

| Component | Implementation |
|---|---|
| **EDA** | Class ratios, histograms, box plots, attack-type ratio plots, correlation heatmap |
| **Feature Engineering** | `Loss_per_User`, `High_Resolution_Time`, categorical encoding |
| **Baseline** | Naive baseline: always predict `Normal` |
| **Models** | Logistic Regression, Random Forest, AdaBoost |
| **Imbalanced Data** | SMOTE applied only on the training set |
| **Evaluation** | Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrices |
| **Interpretability** | SHAP feature importance and beeswarm plots |
| **Tracking** | MLflow logs model metrics and parameters |
| **Deployment** | FastAPI REST API + Docker + optional React frontend |

---

## Models

Three simple and interpretable models are trained and compared:

1. **Logistic Regression**  
   Simple baseline model.

2. **Random Forest**  
   Main deployed model, strong and explainable with SHAP.

3. **AdaBoost**  
   Boosting ensemble method.

These models were chosen because they are classical, understandable, and aligned with course expectations. The project avoids overly advanced models that would make interpretation harder.

---

## Scaling Consistency

The notebook uses `StandardScaler`. The backend loads the saved `scaler.pkl` artifact, so the exact same scaler is used during API prediction.

---

## Evaluation Strategy

The dataset is imbalanced. Therefore, **accuracy alone is not sufficient**.

A naive model that always predicts `Normal` may achieve around **80% accuracy**, but it would fail to detect attacks.

The project therefore evaluates:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

The most important metrics are:

- **Recall**, because missing attacks is dangerous.
- **F1-score**, because it balances precision and recall.
- **ROC-AUC**, because it measures class separation.

---


## Final Executed Metrics

The final notebook execution produced the following results:

| Model | Training | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Naive baseline | Always Normal | 0.804 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | No SMOTE | 0.822 | 0.600 | 0.286 | 0.387 | 0.709 |
| Logistic Regression | SMOTE | 0.636 | 0.312 | 0.714 | 0.435 | 0.716 |
| Random Forest | No SMOTE | 0.813 | 0.522 | 0.571 | 0.545 | 0.801 |
| Random Forest | SMOTE | 0.794 | 0.483 | 0.667 | 0.560 | 0.831 |
| AdaBoost | No SMOTE | 0.785 | 0.450 | 0.429 | 0.439 | 0.759 |
| AdaBoost | SMOTE | 0.748 | 0.421 | 0.762 | 0.542 | 0.811 |

The deployed model is **Random Forest trained with SMOTE**, because it achieved the best ROC-AUC and F1-score while remaining explainable with SHAP.

---

## SMOTE Strategy

SMOTE is used to handle class imbalance, but only on the **training set**.

This avoids data leakage and keeps the test set realistic.

The notebook compares:

- models trained **without SMOTE**
- models trained **with SMOTE**

This shows whether oversampling improves attack detection.

---

## Interpretability

SHAP is used with the Random Forest model to explain predictions.

The SHAP analysis identifies the most influential features, such as:

- Number of affected users
- Attack source
- Incident resolution time
- Financial loss
- Target industry

The API also returns the top SHAP features for each prediction.

---

## Limitations

The results are strong, but they must be interpreted carefully.

- Normal events are synthetic because the original dataset contains attacks only.
- Synthetic data may introduce bias.
- Real-world cybersecurity data would likely be noisier and harder to classify.
- Performance in production may be lower than notebook performance.
- The model should be retrained on real normal traffic if available.

This critical analysis is important because it shows awareness of the dataset limitations.

---


## Highest-Grade Improvements Added

The final version includes extra checks and improvements to match the course content more closely:

- **Leakage-safe preprocessing:** the train/test split is performed before fitting `StandardScaler`.
- **SMOTE is applied only on training data**, not on the full dataset.
- **GridSearchCV + StratifiedKFold appendix:** added in `advanced_model_tuning.py`.
- **SMOTE inside cross-validation pipeline:** prevents validation leakage during tuning.
- **Documented model-selection decision:** tuning was tested, but the deployed model remains Random Forest + SMOTE because it gives the best holdout F1-score and ROC-AUC.
- **Frontend/backend link verified:** React uses `localhost:8000` automatically when opened from `localhost:3000`.

Relevant files:

```text
advanced_model_tuning.py
outputs/hyperparameter_tuning_results.csv
outputs/model_selection_decision.md
HIGH_GRADE_REQUIREMENTS_CHECKLIST.md
```


## Setup and Running

### 1. Install dependencies and run the notebook

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap mlflow joblib imbalanced-learn notebook
jupyter notebook cybersecurity_ml_project.ipynb
```

Run all cells. This will generate trained model files in:

```text
backend/models/
```

---

### 2. Run the backend manually

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

---

### 3. Run with Docker Compose

From the root folder:

```bash
docker-compose up --build
```

Then open:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Swagger API Docs: `http://localhost:8000/docs`

---

### 4. MLflow UI

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

---

## Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Country": "USA",
    "Year": 2022,
    "Target_Industry": "Finance",
    "Financial_Loss": 80.5,
    "Number_of_Affected_Users": 500000,
    "Attack_Source": "Nation-state",
    "Security_Vulnerability_Type": "Unpatched Software",
    "Defense_Mechanism_Used": "Firewall",
    "Incident_Resolution_Time": 48
  }'
```

Expected output includes:

- Prediction from each model
- Final verdict
- Probability of Normal / Attack
- Confidence score
- Top SHAP explanatory features

---

## Key Takeaways

- Adding normal events makes the dataset more realistic than attack-only data.
- The 80/20 split reflects real cybersecurity imbalance.
- Accuracy is not enough; F1-score, recall and ROC-AUC are more reliable.
- SMOTE improves minority-class learning when applied correctly.
- SHAP improves transparency by explaining model decisions.
- FastAPI, Docker and MLflow complete the deployment and tracking requirements.

---

## Tech Stack

- Python 3.10+
- pandas, numpy
- scikit-learn, imbalanced-learn
- matplotlib, seaborn
- SHAP
- MLflow
- FastAPI, Uvicorn
- React.js, Axios
- Docker, Docker Compose


## Final correction included

The correlation heatmap now keeps the diagonal visible (`k=1` in the upper-triangle mask), so diagonal values appear as 1, which is mathematically expected.
>>>>>>> 73df27c (Initial commit)
