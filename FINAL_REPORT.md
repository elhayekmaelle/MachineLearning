# Final Report — Cybersecurity Event Detection  
## Normal vs Attack Classification

**Course:** MOD10 – Machine Learning, Winter 2026  
**Instructor:** Mohammed A. Shehab  
**Dataset:** Global Cybersecurity Threats (2015–2024), adapted with normal events

---

## 1. Project Objective

The objective of this project is to build a complete Machine Learning pipeline for cybersecurity event detection. The model predicts whether an event is **Normal** or an **Attack**.

The original dataset contains cybersecurity attack records only. However, a realistic cybersecurity detection system must compare suspicious activity with normal activity. Therefore, normal events were added to create a binary classification problem.

The final objective is:

\[
\text{Predict event class} = \{\text{Normal}, \text{Attack}\}
\]

This project follows the full model-building lifecycle:

1. Data preprocessing  
2. Exploratory Data Analysis  
3. Feature engineering  
4. Model training and comparison  
5. Evaluation using multiple metrics  
6. Interpretability using SHAP  
7. Deployment with FastAPI and Docker  
8. Experiment tracking with MLflow  

---

## 2. Dataset Preparation

The original dataset contains global cybersecurity threat records from 2015 to 2024. Since the dataset only contains attack cases, normal events were generated and added to make the problem realistic.

A reduced dataset of about **531 events** was used. This follows the project guidance to avoid using unnecessary large amounts of data and to focus on quality of analysis.

The final dataset distribution is approximately:

| Class | Ratio |
|---|---:|
| Normal | 80% |
| Attack | 20% |

This imbalance is intentional because in real cybersecurity environments, attacks are much less frequent than normal activity.

The target variable is `Label`, where:

| Label | Meaning |
|---|---|
| 0 | Normal |
| 1 | Attack |

The column `Attack Type` was removed from model training because it directly describes attack categories and would create target leakage. The label is also not used as a feature.

---

## 3. Exploratory Data Analysis

The EDA focused on ratios rather than raw counts, because the professor specifically requested ratios. The class distribution plot confirms the expected 80/20 imbalance.

Several visualizations were created:

- Class distribution ratio plot
- Normal vs Attack pie chart
- Histograms by class
- Box plots by class
- Attack type ratio chart
- Correlation heatmap

The histograms and box plots show that attack events tend to have higher financial loss, more affected users and longer resolution time. However, the distributions overlap, which is important because a completely separated dataset would make the classification task unrealistically easy.

The attack type ratio chart shows the distribution of attacks among the attack class only. This plot is not used as a training feature; it is used only for EDA.

---

## 4. Correlation Analysis

A correlation heatmap was used to check relationships between numerical and encoded features.

The target label was included only for EDA interpretation. It was not included in the feature matrix.

The analysis checked for high inter-feature correlation. In the course, high correlations such as 0.7 or above are considered problematic because they may indicate redundant variables.

No problematic inter-feature correlation above 0.7 was observed. Therefore, no major feature redundancy problem was detected.

This supports keeping the selected features for model training.

---

## 5. Feature Engineering

Several features were created or transformed to improve model performance.

### 5.1 Loss per User

A new ratio feature was created:

\[
\text{Loss per User} = \frac{\text{Financial Loss}}{\text{Number of Affected Users} + 1}
\]

This is more informative than using only raw financial loss, because it relates the loss to the number of affected users.

### 5.2 High Resolution Time

A binary feature was created to identify events with unusually long incident resolution time:

\[
\text{High Resolution Time} =
\begin{cases}
1, & \text{if resolution time is above median} \\
0, & \text{otherwise}
\end{cases}
\]

### 5.3 Encoding

Categorical variables were encoded using label encoding. The label encoders were saved so that the backend API can apply the same transformations during prediction.

### 5.4 Scaling

Numerical features were scaled using `StandardScaler`. The scaler was also saved for deployment. The backend does not recreate a scaler manually; it loads the saved `scaler.pkl` artifact, ensuring that training and deployment use the same transformation.

---

## 6. Baseline Model

A naive baseline was used before training real models.

The baseline always predicts the majority class:

\[
\text{Prediction} = \text{Normal}
\]

Since the dataset contains approximately 80% normal events, this baseline can reach about 80% accuracy.

However, this model is not useful because it fails to detect attacks. This shows why accuracy alone is misleading for imbalanced datasets.

Therefore, the project focuses on:

- Recall
- F1-score
- ROC-AUC

These metrics are more meaningful for attack detection.

---

## 7. Handling Imbalanced Data

The dataset is intentionally imbalanced with 80% normal events and 20% attacks.

This imbalance is realistic, but it can cause models to favor the majority class.

To handle this issue, SMOTE was applied to the training set only. This is important because applying SMOTE before splitting would cause data leakage. In the improved final notebook, the train/test split is also performed before fitting the `StandardScaler`, so the scaler learns only from the training set and not from the test set.

The notebook compares model performance:

- without SMOTE
- with SMOTE

This comparison shows whether oversampling improves attack detection.

The test set remains imbalanced and realistic.

---

## 8. Model Training and Comparison

Three classical machine learning models were trained and compared.

| Model | Role |
|---|---|
| Logistic Regression | Simple baseline model |
| Random Forest | Main selected model |
| AdaBoost | Boosting ensemble model |

These models were chosen because they are simple enough to explain, but strong enough for a cybersecurity detection task.

Random Forest was selected as the deployed model because it gave the best final balance in the executed notebook: Accuracy = 0.794, Precision = 0.483, Recall = 0.667, F1-score = 0.560 and ROC-AUC = 0.831. It is also compatible with SHAP TreeExplainer, which supports the interpretability requirement.

---

## 9. Evaluation Metrics

The following metrics were calculated:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Accuracy is reported, but it is not the main metric because of class imbalance.

For this project:

- **Recall** is important because missing attacks is dangerous.
- **Precision** is important because too many false alerts reduce trust.
- **F1-score** balances precision and recall.
- **ROC-AUC** evaluates how well the model separates normal and attack events.

The table below contains the actual metrics obtained from the final notebook execution.

| Model | Training | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Naive baseline | Always Normal | 0.804 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | No SMOTE | 0.822 | 0.600 | 0.286 | 0.387 | 0.709 |
| Logistic Regression | SMOTE | 0.636 | 0.312 | 0.714 | 0.435 | 0.716 |
| Random Forest | No SMOTE | 0.813 | 0.522 | 0.571 | 0.545 | 0.801 |
| Random Forest | SMOTE | 0.794 | 0.483 | 0.667 | 0.560 | 0.831 |
| AdaBoost | No SMOTE | 0.785 | 0.450 | 0.429 | 0.439 | 0.759 |
| AdaBoost | SMOTE | 0.748 | 0.421 | 0.762 | 0.542 | 0.811 |

The Random Forest model trained with SMOTE was selected as the deployed model because it provides the best overall ROC-AUC (0.831) and the best F1-score (0.560), while remaining compatible with SHAP TreeExplainer for interpretability.


---


### Hyperparameter Tuning and Cross-Validation Improvement

To strengthen the project and align it with the Class 05 material, an additional hyperparameter tuning appendix was added using **GridSearchCV** and **StratifiedKFold cross-validation**. The tuning focuses on the Random Forest model because Random Forest is an ensemble method and is suitable for imbalanced cybersecurity classification.

The tuning script is available in:

```text
advanced_model_tuning.py
```

It tests several Random Forest parameters, including:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `max_features`

A key improvement is that **SMOTE is placed inside the cross-validation pipeline** using `imblearn.pipeline.Pipeline`. This prevents data leakage because synthetic minority samples are generated only inside the training folds, not before validation.

The tuning results are saved in:

```text
outputs/hyperparameter_tuning_results.csv
outputs/model_selection_decision.md
```

The tuning experiment showed that the tuned Random Forest reached a mean cross-validation ROC-AUC of approximately **0.738**. However, the final deployed Random Forest + SMOTE model from the executed notebook remains the best deployment choice because it achieved stronger holdout performance, especially **F1-score = 0.560** and **ROC-AUC = 0.831**. This is a stronger methodological decision than simply choosing the most complex model: tuning was tested, but the final model was selected based on objective metrics.


## 10. Confusion Matrix Analysis

The confusion matrices show the number of:

- True Normal predictions
- False Alarms
- Missed Attacks
- Correct Attack detections

A good cybersecurity model should minimize missed attacks while keeping false positives reasonable.

The Random Forest model provides a strong balance between detecting attacks and avoiding too many false alarms.

---

## 11. ROC-AUC Analysis

ROC-AUC curves were plotted for all three models.

The ROC-AUC score measures the ability of the model to distinguish between Normal and Attack events.

A score close to 1 indicates strong separation. The Random Forest and AdaBoost models achieved strong ROC-AUC values, showing that they can separate the two classes effectively.

---

## 12. Interpretability with SHAP

SHAP was used to explain the Random Forest model.

The SHAP feature importance plot shows which features have the largest impact on predictions. The most influential features are typically:

- Number of affected users
- Attack source
- Incident resolution time
- Financial loss
- Target industry

The SHAP beeswarm plot also shows the direction of impact. For example, higher values of some risk-related features can increase the model output toward the Attack class.

This interpretability step improves transparency and satisfies the project requirement for explainability.

---

## 13. Deployment

The model is deployed using a FastAPI REST API.

The API includes the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Health check |
| `/options` | GET | Returns valid categorical options |
| `/predict` | POST | Returns Normal/Attack prediction |

The backend loads:

- trained Random Forest model
- Logistic Regression model
- AdaBoost model
- scaler
- label encoders
- feature columns
- SHAP explainer

The API returns:

- predictions from the three models
- final Random Forest verdict
- Normal and Attack probabilities
- confidence score
- top SHAP explanatory features

Docker is used to containerize the backend and frontend.

---

## 14. MLflow Tracking

MLflow is used to track experiments.

For each model, the project logs:

- model name
- training type
- accuracy
- precision
- recall
- F1-score
- ROC-AUC

This allows model versions and metrics to be compared clearly.

---

## 15. Critical Analysis and Limitations

Although the model achieves strong performance, the results should be interpreted with caution.

The main limitation is that normal events are synthetic. They were added because the original dataset contains only attack records. This improves realism compared with using attack-only data, but it may still introduce bias.

The normal and attack distributions were intentionally made overlapping to avoid an unrealistically easy classification problem. However, real cybersecurity data would likely contain more noise, more missing values and more complex patterns.

Therefore, the model is suitable as a complete academic ML pipeline, but production use would require validation on real normal traffic and real attack logs.

---

## 16. Conclusion

This project successfully implements a complete Machine Learning pipeline for cybersecurity event detection.

It includes:

- realistic binary classification
- imbalanced dataset handling
- EDA with ratios and visualizations
- feature engineering
- baseline comparison
- three classical ML models
- evaluation using appropriate metrics
- SHAP interpretability
- MLflow experiment tracking
- FastAPI and Docker deployment

The project satisfies the instructor’s requirements and provides a clear, explainable and deployable cybersecurity detection system.
