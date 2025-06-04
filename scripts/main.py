import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Input, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os

# Load dataset
def load_data(path="data/energydata_complete.csv"):
    df = pd.read_csv(path)
    print("✅ Data loaded.")
    return df

# Preprocess dataset
def preprocess_data(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df = df.drop(columns=["date", "rv1", "rv2"])
    for col in df.columns:
        if col != "Appliances":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.astype({col: 'float64' for col in df.columns if col != "Appliances"})
    print("✅ Preprocessing complete.")
    return df

# Train XGBoost
def train_xgboost(df):
    X = df.drop(columns=["Appliances"])
    y = df["Appliances"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n📊 XGBoost Results:\nMAE: {mean_absolute_error(y_test, y_pred):.2f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    joblib.dump(model, "models/xgboost_model.pkl")
    return model, X_train

# Train LightGBM
def train_lightgbm(df):
    X = df.drop(columns=["Appliances"])
    y = df["Appliances"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n📊 LightGBM Results:\nMAE: {mean_absolute_error(y_test, y_pred):.2f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    joblib.dump(model, "models/lightgbm_model.pkl")
    return model, X_train

# SHAP explanation
def explain_with_shap(model, X_train, model_name="model"):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(f"shap_summary_{model_name}.png")
    plt.clf()
    print(f"Saved SHAP summary: shap_summary_{model_name}.png")

# LIME explanation
def explain_with_lime(model, X_train, index=0, model_name="model"):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        mode="regression"
    )
    instance = X_train.iloc[index].values
    explanation = explainer.explain_instance(instance, model.predict)
    explanation.save_to_file(f"lime_explanation_{model_name}_{index}.html")
    print(f"LIME explanation saved: lime_explanation_{model_name}_{index}.html")

# Sequence preparation
def prepare_sequence_data(df, target_column="Appliances", sequence_length=24):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i + sequence_length])
        y.append(df.iloc[i + sequence_length][target_column])
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:], scaler

# LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Train LSTM
def train_lstm(X_train, y_train, X_test, y_test):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=15, batch_size=32, verbose=1)
    y_pred = model.predict(X_test)
    print(f"\n📊 LSTM Results:\nMAE: {mean_absolute_error(y_test, y_pred):.2f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    model.save("models/lstm_model.keras")
    return model


# TCN model
def build_tcn_model(input_shape):
    model = Sequential([
        Conv1D(128, kernel_size=3, padding='causal', activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Conv1D(64, kernel_size=3, padding='causal', activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Train TCN
def train_tcn(X_train, y_train, X_test, y_test):
    model = build_tcn_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=15, batch_size=32, verbose=1)
    y_pred = model.predict(X_test)
    print(f"\n📊 TCN Results:\nMAE: {mean_absolute_error(y_test, y_pred):.2f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    model.save("models/tcn_model.keras")
    return model

# Main
if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)

    # Tree models
    model_xgb, X_train_xgb = train_xgboost(df)
    explain_with_shap(model_xgb, X_train_xgb, model_name="xgboost")
    explain_with_lime(model_xgb, X_train_xgb, 0, "xgboost")

    model_lgb, X_train_lgb = train_lightgbm(df)
    explain_with_shap(model_lgb, X_train_lgb, model_name="lightgbm")
    explain_with_lime(model_lgb, X_train_lgb, 0, "lightgbm")

    # Sequence models
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, _ = prepare_sequence_data(df)
    lstm_model = train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    tcn_model = train_tcn(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
