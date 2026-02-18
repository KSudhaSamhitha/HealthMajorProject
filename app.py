import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

st.title("🫀 Multimodal Explainable AI for Heart Disease Risk Prediction")

# Load model
MODEL_PATH = "outputs/results/Tuned_XGBoost.pkl"
model = joblib.load(MODEL_PATH)

# Load scaler columns
X_columns = pd.read_csv("processed_data/X_scaled.csv").columns

st.sidebar.header("Patient Clinical & Lifestyle Inputs")

def user_input():
    data = {}
    for feature in X_columns:
        data[feature] = st.sidebar.number_input(feature, value=0.0)
    return pd.DataFrame([data])

input_df = user_input()

threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.2)

if st.button("Predict Risk"):

    probability = model.predict_proba(input_df)[0][1]
    prediction = 1 if probability >= threshold else 0

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease")
    else:
        st.success(f"✅ Low Risk")

    st.write(f"Predicted Probability: {probability:.4f}")
    st.write(f"Threshold Used: {threshold}")

    # SHAP Explanation
    st.subheader("🔍 SHAP Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        shap_vals = shap_values[0]
        base_value = explainer.expected_value

    fig = plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=input_df.iloc[0],
            feature_names=X_columns
        ),
        show=False
    )

    st.pyplot(fig)
