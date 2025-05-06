# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Load preprocessed data (optional: to extract feature ranges)
def load_sample_data():
    df = pd.read_csv("data/energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df = df.drop(columns=["date", "rv1", "rv2"])
    for col in df.columns:
        if col != "Appliances":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.astype({col: 'float64' for col in df.columns if col != "Appliances"})
    return df

# Load model
@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgboost_model.pkl")
    lgb_model = joblib.load("models/lightgbm_model.pkl")
    return xgb_model, lgb_model

# SHAP explanation
@st.cache_resource
def get_shap_explainer(_model, _X):
    explainer = shap.Explainer(_model, _X)
    return explainer


# Streamlit UI
st.set_page_config(page_title="Energy Forecasting Dashboard", layout="wide")
st.title("🔌 Explainable AI for Appliance Energy Forecasting")

sample_df = load_sample_data()
X = sample_df.drop(columns=["Appliances"])
xgb_model, lgb_model = load_models()

# Model selector
model_choice = st.selectbox("Choose a model to predict:", ["XGBoost", "LightGBM"])
model = xgb_model if model_choice == "XGBoost" else lgb_model

# Sidebar sliders for input features
st.sidebar.header("Input Features")
user_input = {}
for col in X.columns:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())
    user_input[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=mean_val)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Make prediction
prediction = model.predict(input_df)[0]
st.subheader(f"⚡ Predicted Appliance Energy Usage: {prediction:.2f} Wh")

# SHAP Explanation
with st.expander("🔍 SHAP Explanation"):
    explainer = get_shap_explainer(model, X)
    shap_values = explainer(input_df)
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches="tight")

st.caption("Built using Streamlit, SHAP, XGBoost and LightGBM")
