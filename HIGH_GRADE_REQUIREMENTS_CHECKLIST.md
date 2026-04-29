# High-Grade Requirements Checklist

This file summarizes how the final project addresses the official MOD10 ML project requirements and the concepts covered in class.

| Requirement | Status | Evidence in project |
|---|---:|---|
| Data preprocessing | Complete | Missing values/feature preparation, encoding, scaling |
| Leakage prevention | Improved | Train/test split before scaler fitting; `Attack Type` removed from features |
| EDA | Complete | Histograms, boxplots, heatmap, class ratio plots in `outputs/` |
| Feature engineering | Complete | `Loss_per_User`, `High_Resolution_Time` |
| At least two models | Complete | Logistic Regression, Random Forest, AdaBoost |
| Ensemble methods | Complete | Random Forest and AdaBoost |
| Cross-validation | Added | `advanced_model_tuning.py` with StratifiedKFold |
| Hyperparameter tuning | Added | GridSearchCV for Random Forest |
| Imbalanced data handling | Complete | SMOTE on training data; SMOTE inside CV pipeline for tuning |
| Metrics | Complete | Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices |
| Interpretability | Complete | SHAP feature importance and API top features |
| MLflow tracking | Complete | MLflow experiment database and logging in notebook |
| Docker deployment | Complete | Backend and frontend Dockerfiles + `docker-compose.yml` |
| REST API | Complete | FastAPI `/predict`, `/options`, `/health` |
| Frontend/backend link | Complete | React connects to `localhost:8000` from `localhost:3000` |
| Final report | Complete | `FINAL_REPORT.md` includes methodology, results, limitations, deployment |

## Final model decision

The project includes hyperparameter tuning for a stronger methodology. The tuned model is documented, but the deployed model remains Random Forest + SMOTE because it had the best final balance in the executed notebook:

- F1-score: 0.560
- ROC-AUC: 0.831

This is a good ML decision because the final choice is based on evaluation metrics, not only model complexity.
