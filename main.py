import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam
import joblib
import os

print("✅ Data loaded.")

# Load and preprocess data
def load_data():
    df = pd.read_csv("data/energydata_complete.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df.drop(columns=["date", "rv1", "rv2"], inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Feature Engineering
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df.dropna(inplace=True)
    return df

df = load_data()
print("✅ Preprocessing complete.")

# XGBoost Training
def train_xgboost(df):
    features = df.drop(columns=["Appliances"])
    target = df["Appliances"]
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.1, random_state=42)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print("\n📊 XGBoost Results:")
    print("MAE:", round(mean_absolute_error(y_val, y_pred), 2), "Wh")
    print("RMSE:", round(np.sqrt(mean_squared_error(y_val, y_pred)), 2), "Wh")
    joblib.dump(model, "models/xgboost_model.pkl")
    return model, X_train

# LightGBM Training
def train_lightgbm(df):
    features = df.drop(columns=["Appliances"])
    target = df["Appliances"]
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.1, random_state=42)

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print("\n📊 LightGBM Results:")
    print("MAE:", round(mean_absolute_error(y_val, y_pred), 2), "Wh")
    print("RMSE:", round(np.sqrt(mean_squared_error(y_val, y_pred)), 2), "Wh")
    joblib.dump(model, "models/lightgbm_model.pkl")
    return model, X_train

# LSTM Model
def train_lstm(df):
    features = df.drop(columns=["Appliances"])
    target = df["Appliances"]

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, "models/lstm_scaler.pkl")

    X_seq, y_seq = [], []
    window_size = 10
    for i in range(len(features_scaled) - window_size):
        X_seq.append(features_scaled[i:i+window_size])
        y_seq.append(target.iloc[i + window_size])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)

    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X_train, y_train, epochs=32, batch_size=48, validation_data=(X_val, y_val), verbose=1)
    model.save("models/lstm_model.keras")

    y_pred = model.predict(X_val).flatten()
    print("\n📊 LSTM Results:")
    print("MAE:", round(mean_absolute_error(y_val, y_pred), 2), "Wh")
    print("RMSE:", round(np.sqrt(mean_squared_error(y_val, y_pred)), 2), "Wh")

# TCN Model
def train_tcn(df):
    features = df.drop(columns=["Appliances"])
    target = df["Appliances"]

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, "models/tcn_scaler.pkl")

    X_seq, y_seq = [], []
    window_size = 10
    for i in range(len(features_scaled) - window_size):
        X_seq.append(features_scaled[i:i+window_size])
        y_seq.append(target.iloc[i + window_size])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=3, padding="causal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv1D(filters=64, kernel_size=3, padding="causal"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X_train, y_train, epochs=32, batch_size=48, validation_data=(X_val, y_val), verbose=1)
    model.save("models/tcn_model.keras")

    y_pred = model.predict(X_val).flatten()
    print("\n📊 TCN Results:")
    print("MAE:", round(mean_absolute_error(y_val, y_pred), 2), "Wh")
    print("RMSE:", round(np.sqrt(mean_squared_error(y_val, y_pred)), 2), "Wh")

# Run training for all models
os.makedirs("models", exist_ok=True)
model_xgb, X_train_xgb = train_xgboost(df)
model_lgb, X_train_lgb = train_lightgbm(df)
train_lstm(df)
train_tcn(df)
