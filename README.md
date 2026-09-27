# Credit Card Fraud Detection — End-to-End ML Pipeline

## Business problem
Card issuers lose billions annually to fraudulent transactions, but fraud
is extremely rare (~0.17% of transactions here) — a naive model that
predicts "not fraud" every time is 99.8% "accurate" and completely
useless. This project builds a classifier that catches fraud while
staying usable in production, and explains *why* it flags each
transaction.

## Data
[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284,807 anonymized European card transactions, 492 confirmed frauds.
Features V1–V28 are PCA components (anonymized for privacy); `Time` and
`Amount` are raw.

## Approach
1. **EDA**: confirmed extreme class imbalance, examined amount/time
   patterns by class.
2. **Preprocessing**: scaled `Amount`/`Time` (PCA components already
   standardized); stratified train/test split to preserve fraud ratio
   in both sets.
3. **Imbalance handling**: compared SMOTE (oversampling) with XGBoost's
   native `scale_pos_weight` — avoided the common mistake of applying
   SMOTE before the train/test split (which leaks synthetic fraud
   patterns into the test set).
4. **Modeling**: Logistic Regression baseline vs XGBoost.
5. **Metric choice**: prioritized **PR-AUC over ROC-AUC**, since ROC-AUC
   is misleadingly optimistic under heavy class imbalance.
6. **Threshold tuning**: chose a decision threshold targeting ≥85%
   recall, reflecting that missed fraud (false negative) costs the
   business far more than a false alarm (false positive).
7. **Explainability**: SHAP values show which features drive each
   prediction — critical for a fraud team that needs to justify
   blocking a transaction, not just a black-box score.
8. **Deployment**: Streamlit app for both batch CSV scoring and
   single-transaction scoring with a live SHAP explanation.

## Results
*(fill in your actual numbers after running `eda_and_modeling.py`)*

Model	ROC-AUC	PR-AUC	Recall @ chosen threshold
Logistic Regression	0.9698	0.7249	91.84%
XGBoost	0.9797	0.8634	85.71% (threshold = 0.327)
## What I'd improve with more time
- Try cost-sensitive learning instead of a fixed threshold
- Add time-based validation (fraud patterns drift over time — a random
  split can overstate real-world performance)
- Feature engineering on transaction velocity if raw (non-PCA) data
  were available

## How to run
```bash
pip install -r requirements.txt
mkdir -p data outputs
# place creditcard.csv inside data/
python eda_and_modeling.py
streamlit run app.py
```

## Live demo
*(add your deployed Streamlit Community Cloud / Hugging Face Spaces link here)*
