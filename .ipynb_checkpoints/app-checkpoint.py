"""
Streamlit app for the fraud detection model.
Run locally with:  streamlit run app.py
Deploy free at: https://share.streamlit.io (Streamlit Community Cloud)
or https://huggingface.co/spaces (Hugging Face Spaces, Streamlit SDK)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/xgb_fraud_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    feature_cols = joblib.load("outputs/feature_columns.pkl")
    threshold = joblib.load("outputs/chosen_threshold.pkl")
    return model, scaler, feature_cols, threshold

model, scaler, feature_cols, threshold = load_artifacts()
explainer = shap.TreeExplainer(model)

st.title("💳 Credit Card Fraud Detection")
st.write(
    "Upload transactions (same schema as the Kaggle creditcard.csv, "
    "minus the Class column) or score a single transaction below."
)

tab1, tab2 = st.tabs(["Upload CSV", "Single Transaction"])

# ---------------- TAB 1: Batch scoring ----------------
with tab1:
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        data["Amount_scaled"] = scaler.fit_transform(data[["Amount"]])
        data["Time_scaled"] = scaler.fit_transform(data[["Time"]])
        X = data[feature_cols]

        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)

        result = data.copy()
        result["fraud_probability"] = probs
        result["flagged_as_fraud"] = preds

        st.write(f"Flagged {preds.sum()} out of {len(preds)} transactions "
                 f"at threshold {threshold:.3f}.")
        st.dataframe(result.sort_values("fraud_probability", ascending=False).head(50))

        st.download_button(
            "Download scored results",
            result.to_csv(index=False),
            file_name="scored_transactions.csv",
        )

# ---------------- TAB 2: Single transaction with SHAP explanation ----------------
with tab2:
    st.write("Enter values manually (defaults are dataset means) or pick a row index from your test set.")
    col1, col2 = st.columns(2)
    inputs = {}
    for i, col in enumerate(feature_cols):
        target_col = col1 if i % 2 == 0 else col2
        inputs[col] = target_col.number_input(col, value=0.0, format="%.4f")

    if st.button("Score transaction"):
        X_single = pd.DataFrame([inputs])[feature_cols]
        prob = model.predict_proba(X_single)[:, 1][0]
        pred = int(prob >= threshold)

        st.metric("Fraud probability", f"{prob:.2%}")
        st.write("🚨 Flagged as FRAUD" if pred == 1 else "✅ Looks legitimate")

        # SHAP explanation for this single prediction
        shap_values = explainer.shap_values(X_single)
        fig, ax = plt.subplots()
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_single.iloc[0],
                feature_names=feature_cols,
            ),
            show=False,
        )
        st.pyplot(fig)
        st.caption(
            "Bars show which features pushed this prediction toward "
            "fraud (red) or legitimate (blue)."
        )

st.divider()
st.caption(
    "Model: XGBoost trained on the Kaggle Credit Card Fraud dataset. "
    "Threshold chosen to prioritize recall (catching fraud) over precision, "
    "reflecting the higher business cost of missed fraud vs false alarms."
)
