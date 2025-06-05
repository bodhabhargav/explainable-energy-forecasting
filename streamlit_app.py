import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load data
def load_data():
    df = pd.read_csv("data/energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df.drop(columns=["date", "rv1", "rv2"], inplace=True)
    df = df.astype({col: 'float64' for col in df.columns})
    df.dropna(inplace=True)
    return df

@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgboost_model.pkl")
    lgb_model = joblib.load("models/lightgbm_model.pkl")
    lstm_model = load_model("models/lstm_model.keras", compile=False)
    tcn_model = load_model("models/tcn_model.keras", compile=False)
    return xgb_model, lgb_model, lstm_model, tcn_model

@st.cache_data
def get_scaler(model_name):
    return joblib.load(f"models/{model_name}_scaler.pkl")

# Prepare LSTM/TCN input

def prepare_sequence_input(df, user_input, model_name, seq_len=10):
    scaler = get_scaler(model_name)
    df_copy = df.drop(columns=["Appliances"]).copy()
    user_df = pd.DataFrame([user_input])
    full_df = pd.concat([df_copy, user_df], ignore_index=True)
    scaled = scaler.fit_transform(full_df)
    return np.array(scaled[-seq_len:]).reshape(1, seq_len, -1)

# Initialize app
st.set_page_config(page_title="Energy Forecasting", layout="wide")
st.title("🔌 Explainable AI for Appliance Energy Forecasting")

# Load data & models
df = load_data()
xgb_model, lgb_model, lstm_model, tcn_model = load_models()
X = df.drop(columns=["Appliances"])

# Sidebar - User input
st.sidebar.header("🔧 Input Features")
model_choice = st.sidebar.selectbox("Choose a model:", ["XGBoost", "LightGBM", "LSTM", "TCN"])
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

input_df = pd.DataFrame([input_data])

st.write("### 🔍 Input Preview")
st.dataframe(input_df)

# Prediction & Explanation
if st.button("Predict Energy Usage"):
    if model_choice == "XGBoost":
        prediction = xgb_model.predict(input_df)[0]
        st.subheader(f"⚡ XGBoost Predicted Usage: {prediction:.2f} Wh")

        explainer = shap.Explainer(xgb_model, X)
        shap_values = explainer(input_df)
        st.write("#### SHAP Explanation")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

        st.write("#### LIME Explanation")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            mode='regression')
        lime_exp = lime_explainer.explain_instance(input_df.values[0], xgb_model.predict)
        st.write(dict(lime_exp.as_list()))

    elif model_choice == "LightGBM":
        prediction = lgb_model.predict(input_df)[0]
        st.subheader(f"⚡ LightGBM Predicted Usage: {prediction:.2f} Wh")

        explainer = shap.Explainer(lgb_model, X)
        shap_values = explainer(input_df)
        st.write("#### SHAP Explanation")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

        st.write("#### LIME Explanation")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            mode='regression')
        lime_exp = lime_explainer.explain_instance(input_df.values[0], lgb_model.predict)
        st.write(dict(lime_exp.as_list()))

    elif model_choice == "LSTM":
        seq_input = prepare_sequence_input(df, input_data, "lstm")
        prediction = lstm_model.predict(seq_input)[0][0]
        st.subheader(f"⚡ LSTM Predicted Usage: {prediction:.2f} Wh")

    elif model_choice == "TCN":
        seq_input = prepare_sequence_input(df, input_data, "tcn")
        prediction = tcn_model.predict(seq_input)[0][0]
        st.subheader(f"⚡ TCN Predicted Usage: {prediction:.2f} Wh")

st.caption("Built with Streamlit, SHAP, LIME, XGBoost, LightGBM, LSTM, and TCN")
