"""
Credit Card Fraud Detection - EDA, Modeling & Explainability
================================================================
Dataset: Kaggle "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud)
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place the file as: data/creditcard.csv

284,807 transactions, 492 frauds (0.17%) — real anonymized European
cardholder transactions. Features V1-V28 are PCA components (already
anonymized); Time and Amount are raw.

Run this as a script or paste cells into a Jupyter notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib

RANDOM_STATE = 42

# -----------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------
df = pd.read_csv("data/creditcard.csv")
print(df.shape)
print(df["Class"].value_counts(normalize=True))  # ~0.17% fraud -> extreme imbalance

# -----------------------------------------------------------------
# 2. EDA
# -----------------------------------------------------------------
# Class imbalance plot
sns.countplot(x="Class", data=df)
plt.title("Class distribution (0 = legit, 1 = fraud)")
plt.savefig("outputs/class_distribution.png", bbox_inches="tight")
plt.close()

# Amount distribution: fraud vs legit
plt.figure(figsize=(8, 4))
sns.kdeplot(df[df.Class == 0]["Amount"], label="Legit", clip=(0, 500))
sns.kdeplot(df[df.Class == 1]["Amount"], label="Fraud", clip=(0, 500))
plt.title("Transaction amount distribution by class")
plt.legend()
plt.savefig("outputs/amount_distribution.png", bbox_inches="tight")
plt.close()

# Correlation of top features with target (PCA components are already
# decorrelated by construction, so this mostly confirms which V-features
# separate classes best)
corrs = df.corr()["Class"].abs().sort_values(ascending=False)[1:11]
print("Top 10 features correlated with fraud:\n", corrs)

# WHY: Time and Amount are on totally different scales from the PCA
# components (V1-V28 are already standardized). Tree models don't care,
# but Logistic Regression does -> scale them.
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
df = df.drop(["Amount", "Time"], axis=1)

# -----------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (stratified — critical given the imbalance)
# -----------------------------------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# WHY SMOTE only on training data: applying it before the split would
# leak synthetic copies of test-set fraud cases into training, inflating
# your metrics artificially. This is a very common beginner mistake —
# mentioning you avoided it is a strong interview signal.
sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE:", y_train_res.value_counts().to_dict())

# -----------------------------------------------------------------
# 4. MODEL 1 — Logistic Regression (baseline, interpretable)
# -----------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_res, y_train_res)
y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred_lr, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print("PR-AUC:", average_precision_score(y_test, y_prob_lr))

# -----------------------------------------------------------------
# 5. MODEL 2 — XGBoost (stronger, handles imbalance natively too)
# -----------------------------------------------------------------
# scale_pos_weight lets XGBoost handle imbalance without needing SMOTE;
# training both ways (with/without SMOTE) is a good ablation to mention.
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=RANDOM_STATE,
)
xgb_model.fit(X_train, y_train)  # note: raw imbalanced data + scale_pos_weight
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n--- XGBoost ---")
print(classification_report(y_test, y_pred_xgb, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
print("PR-AUC:", average_precision_score(y_test, y_prob_xgb))

# -----------------------------------------------------------------
# 6. WHY PR-AUC MATTERS MORE THAN ROC-AUC HERE
# -----------------------------------------------------------------
# With 0.17% positive class, ROC-AUC can look great (>0.97) even for a
# mediocre model, because true negatives dominate. Precision-Recall AUC
# is the honest metric for extreme imbalance — say this explicitly in
# your README/interview, it shows real understanding, not just copying
# metrics off a Kaggle notebook.

precision, recall, thresholds = precision_recall_curve(y_test, y_prob_xgb)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — XGBoost")
plt.savefig("outputs/pr_curve.png", bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------
# 7. CONFUSION MATRIX AT A BUSINESS-CHOSEN THRESHOLD
# -----------------------------------------------------------------
# Default 0.5 threshold is usually wrong for fraud. Pick a threshold
# based on the cost tradeoff: missing fraud (false negative) is usually
# far costlier than a false alarm. Example: pick threshold that gives
# recall >= 0.85.
target_recall = 0.85
idx = np.argmax(recall >= target_recall) if np.any(recall >= target_recall) else -1
chosen_threshold = thresholds[max(idx - 1, 0)]
y_pred_business = (y_prob_xgb >= chosen_threshold).astype(int)

print(f"\nChosen threshold for recall>={target_recall}: {chosen_threshold:.4f}")
print(confusion_matrix(y_test, y_pred_business))
print(classification_report(y_test, y_pred_business, digits=4))

# -----------------------------------------------------------------
# 8. EXPLAINABILITY WITH SHAP
# -----------------------------------------------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test.iloc[:1000])  # sample for speed

shap.summary_plot(shap_values, X_test.iloc[:1000], show=False)
plt.savefig("outputs/shap_summary.png", bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------
# 9. SAVE ARTIFACTS FOR THE STREAMLIT APP
# -----------------------------------------------------------------
joblib.dump(xgb_model, "outputs/xgb_fraud_model.pkl")
joblib.dump(scaler, "outputs/scaler.pkl")
joblib.dump(list(X.columns), "outputs/feature_columns.pkl")
joblib.dump(chosen_threshold, "outputs/chosen_threshold.pkl")

print("\nDone. Model + artifacts saved to outputs/")
