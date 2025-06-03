import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer

# Load sample data
@st.cache_data
def load_data():
    df = pd.read_csv("data/energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df.drop(columns=["date", "rv1", "rv2"], inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.astype({col: 'float64' for col in df.columns})
    df.dropna(inplace=True)
    return df

# Load models
@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgboost_model.pkl")
    lgb_model = joblib.load("models/lightgbm_model.pkl")
    lstm_model = load_model("models/lstm_model.keras", compile=False)
    tcn_model = load_model("models/tcn_model.keras", compile=False)
    return xgb_model, lgb_model, lstm_model, tcn_model

# SHAP explainer
@st.cache_resource
def get_shap_explainer(_model, _X):
    return shap.Explainer(_model, _X)

# Prepare sequence input (ensure 29 features for LSTM/TCN)
def prepare_sequence_input(df, user_input, seq_len=24):
    df_copy = df.copy()
    user_df = pd.DataFrame([user_input])
    df_with_input = pd.concat([df_copy, user_df], ignore_index=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_with_input)

    last_sequence = scaled[-seq_len:]
    return np.array(last_sequence).reshape(1, seq_len, -1)

# Build background sequences for explainers
@st.cache_resource
def get_sequence_background(df, seq_len=24, sample_size=50):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    sequences = []
    for i in range(len(scaled) - seq_len):
        sequences.append(scaled[i:i + seq_len])
    sequences = np.array(sequences)
    if len(sequences) > sample_size:
        idx = np.random.choice(len(sequences), sample_size, replace=False)
        sequences = sequences[idx]
    return sequences

def _flatten(seqs):
    return seqs.reshape(seqs.shape[0], -1)

# --- Streamlit App ---

st.set_page_config(page_title="Energy Forecasting", layout="wide")
st.title("🔌 Explainable AI for Appliance Energy Forecasting")

df = load_data()
xgb_model, lgb_model, lstm_model, tcn_model = load_models()

# Sidebar inputs
st.sidebar.header("🔧 Input Features")
model_choice = st.sidebar.selectbox("Choose a model:", ["XGBoost", "LightGBM", "LSTM", "TCN"])
input_data = {}

for col in df.columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    input_data[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

input_df = pd.DataFrame([input_data])
X = df.drop(columns=["Appliances"])

# Prediction
st.write("### 🔍 Input Preview")
st.dataframe(input_df)

if st.button("Predict Energy Usage"):
    if model_choice in ["XGBoost", "LightGBM"]:
        model = xgb_model if model_choice == "XGBoost" else lgb_model
        prediction = model.predict(input_df.drop(columns=["Appliances"]))[0]
        st.subheader(f"⚡ {model_choice} Predicted Energy Usage: {prediction:.2f} Wh")

        with st.expander("🔍 SHAP Explanation"):
            explainer = get_shap_explainer(model, X)
            shap_values = explainer(input_df.drop(columns=["Appliances"]))
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

    elif model_choice in ["LSTM", "TCN"]:
        seq_input = prepare_sequence_input(df, input_data, seq_len=24)
        model = lstm_model if model_choice == "LSTM" else tcn_model
        prediction = model.predict(seq_input)[0][0]
        st.subheader(f"⚡ {model_choice} Predicted Energy Usage: {prediction:.2f} Wh")

        background = get_sequence_background(df, seq_len=24, sample_size=50)
        background_flat = _flatten(background)
        instance_flat = _flatten(seq_input)

        def _predict(seqs_flat):
            seqs = seqs_flat.reshape(seqs_flat.shape[0], seq_input.shape[1], seq_input.shape[2])
            return model.predict(seqs).reshape(-1)

        feature_names = [f"{col}_t{i}" for i in range(seq_input.shape[1]) for col in df.columns]

        with st.expander("🔍 SHAP Explanation"):
            explainer = shap.KernelExplainer(_predict, background_flat)
            shap_values = explainer(instance_flat)
            explanation = shap.Explanation(values=shap_values[0],
                                           base_values=explainer.expected_value,
                                           data=instance_flat[0],
                                           feature_names=feature_names)
            fig, ax = plt.subplots()
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)

        with st.expander("📋 LIME Explanation"):
            lime_explainer = LimeTabularExplainer(background_flat,
                                                 mode="regression",
                                                 feature_names=feature_names)
            explanation = lime_explainer.explain_instance(instance_flat[0], _predict, num_features=10)
            st.pyplot(explanation.as_pyplot_figure())

st.caption("Built with Streamlit, SHAP, XGBoost, LightGBM, LSTM, and TCN")
