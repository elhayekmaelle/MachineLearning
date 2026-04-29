"""
Advanced model tuning appendix for the MOD10 cybersecurity ML project.

Purpose:
- Demonstrates GridSearchCV / StratifiedKFold, as studied in Class 05.
- Places SMOTE inside the cross-validation pipeline to avoid data leakage.
- Compares the tuned model against the final deployed model decision.

Run from the project root:
    python advanced_model_tuning.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

os.makedirs("outputs", exist_ok=True)

df_attacks_original = pd.read_csv("Global_Cybersecurity_Threats_2015-2024.csv")

TOTAL_SAMPLES = 531
N_ATTACKS = int(TOTAL_SAMPLES * 0.20)
N_NORMAL = TOTAL_SAMPLES - N_ATTACKS

required_cols = [
    "Country", "Year", "Target Industry", "Attack Type",
    "Financial Loss (in Million $)", "Number of Affected Users",
    "Attack Source", "Security Vulnerability Type",
    "Defense Mechanism Used", "Incident Resolution Time (in Hours)"
]

df_attack = df_attacks_original.sample(n=N_ATTACKS, random_state=RANDOM_STATE).copy()
df_attack["Label"] = 1


def clipped_normal(mean, std, size, low, high):
    return np.clip(np.random.normal(mean, std, size), low, high)


df_normal = pd.DataFrame({
    "Country": np.random.choice(df_attacks_original["Country"].dropna().unique(), N_NORMAL),
    "Year": np.random.choice(df_attacks_original["Year"].dropna().unique(), N_NORMAL),
    "Target Industry": np.random.choice(df_attacks_original["Target Industry"].dropna().unique(), N_NORMAL),
    "Attack Type": "None",
    "Financial Loss (in Million $)": clipped_normal(
        df_attack["Financial Loss (in Million $)"].mean() * 0.75,
        max(df_attack["Financial Loss (in Million $)"].std() * 0.90, 10),
        N_NORMAL, 0, 100
    ),
    "Number of Affected Users": clipped_normal(
        df_attack["Number of Affected Users"].mean() * 0.65,
        max(df_attack["Number of Affected Users"].std() * 0.90, 100000),
        N_NORMAL, 0, 1_000_000
    ).astype(int),
    "Attack Source": np.random.choice(df_attacks_original["Attack Source"].dropna().unique(), N_NORMAL),
    "Security Vulnerability Type": np.random.choice(df_attacks_original["Security Vulnerability Type"].dropna().unique(), N_NORMAL),
    "Defense Mechanism Used": np.random.choice(df_attacks_original["Defense Mechanism Used"].dropna().unique(), N_NORMAL),
    "Incident Resolution Time (in Hours)": clipped_normal(
        df_attack["Incident Resolution Time (in Hours)"].mean() * 0.75,
        max(df_attack["Incident Resolution Time (in Hours)"].std() * 0.90, 10),
        N_NORMAL, 1, 72
    ).astype(int),
    "Label": 0
})

df = pd.concat([df_normal, df_attack[required_cols + ["Label"]]], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

df_model = df.drop(columns=["Attack Type"]).copy()
df_model["Loss_per_User"] = (
    df_model["Financial Loss (in Million $)"] /
    (df_model["Number of Affected Users"] + 1)
)
median_resolution = df_model["Incident Resolution Time (in Hours)"].median()
df_model["High_Resolution_Time"] = (
    df_model["Incident Resolution Time (in Hours)"] > median_resolution
).astype(int)

for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col + "_enc"] = le.fit_transform(df_model[col].astype(str))
    df_model.drop(columns=[col], inplace=True)

y = df_model["Label"].astype(int)
X = df_model.drop(columns=["Label"])

# Leakage-safe order:
# 1. split first
# 2. fit scaler only on training data
# 3. apply SMOTE only on training folds / training data
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rf_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("model", RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=1
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [6, 8, None],
    "model__min_samples_leaf": [1, 2],
    "model__max_features": ["sqrt"],
}

grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    refit=True
)
grid.fit(X_train, y_train)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

models = {
    "Naive baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression + SMOTE": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Random Forest tuned + SMOTE-CV": grid.best_estimator_,
    "AdaBoost + SMOTE": AdaBoostClassifier(
        n_estimators=100, learning_rate=0.7, random_state=RANDOM_STATE
    )
}

rows = []
for name, model in models.items():
    if name in ["Logistic Regression + SMOTE", "AdaBoost + SMOTE"]:
        model.fit(X_train_smote, y_train_smote)
    elif name == "Naive baseline":
        model.fit(X_train, y_train)
    # tuned pipeline is already fitted by GridSearchCV

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    })

results = pd.DataFrame(rows)
results.to_csv("outputs/hyperparameter_tuning_results.csv", index=False)

with open("outputs/model_selection_decision.md", "w", encoding="utf-8") as f:
    f.write("# Model Selection Decision After Hyperparameter Tuning\n\n")
    f.write("GridSearchCV was added to align the project with Class 05: cross-validation and model tuning.\n\n")
    f.write("## Best Random Forest parameters\n\n")
    f.write(f"```text\n{grid.best_params_}\n```\n\n")
    f.write(f"Best mean CV ROC-AUC: {grid.best_score_:.3f}\n\n")
    f.write("## Results\n\n")
    f.write(results.round(3).to_markdown(index=False))
    f.write("\n\nThe tuned model is documented as an appendix. The final deployed model can remain the simpler Random Forest + SMOTE if it gives stronger holdout F1/ROC-AUC in the executed notebook.\n")

print("Best parameters:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 3))
print(results.round(3))
print("Saved outputs/hyperparameter_tuning_results.csv")
print("Saved outputs/model_selection_decision.md")
