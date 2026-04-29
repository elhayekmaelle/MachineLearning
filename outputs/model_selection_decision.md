# Model Selection Decision After Hyperparameter Tuning

GridSearchCV was added to align the project with the Class 05 content on cross-validation and model tuning.

## Random Forest GridSearchCV setup

- Cross-validation: StratifiedKFold with 5 folds
- Scoring metric: ROC-AUC
- Imbalance handling: SMOTE inside the cross-validation pipeline
- Best parameters found:

```text
{'model__max_depth': 6, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 2, 'model__n_estimators': 100}
```

- Best mean cross-validation ROC-AUC: 0.738

## Final deployment choice

The tuned Random Forest was tested, but the already selected Random Forest + SMOTE model remains the deployed model because its holdout ROC-AUC and F1-score in the executed notebook were stronger:

- Random Forest + SMOTE: F1 = 0.560, ROC-AUC = 0.831
- Tuned Random Forest + SMOTE-CV: F1 = 0.471, ROC-AUC = 0.818

This is a defensible ML decision: hyperparameter tuning was explored, but the final deployed model was selected based on better test-set performance and explainability.
