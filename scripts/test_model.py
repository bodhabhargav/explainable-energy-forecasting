import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess dataset
def load_data(path="data/energydata_complete.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df = df.drop(columns=["date", "rv1", "rv2"])
    df = df.astype({col: 'float64' for col in df.columns if col != "Appliances"})
    return df

# Prepare sequential data
def prepare_sequence_data(df, target_column="Appliances", sequence_length=24):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i + sequence_length])
        y.append(df.iloc[i + sequence_length][target_column])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Predict with tree models
def predict_tree_models(df):
    X = df.drop(columns=["Appliances"])
    y = df["Appliances"]

    print("\n📦 Testing XGBoost:")
    xgb_model = joblib.load("models/xgboost_model.pkl")
    xgb_pred = xgb_model.predict(X[-1:])[0]
    print(f"XGBoost prediction: {xgb_pred:.2f} Wh")

    print("\n📦 Testing LightGBM:")
    lgb_model = joblib.load("models/lightgbm_model.pkl")
    lgb_pred = lgb_model.predict(X[-1:])[0]
    print(f"LightGBM prediction: {lgb_pred:.2f} Wh")

# Predict with LSTM and TCN
def predict_sequence_models(df):
    X_seq, y_seq, _ = prepare_sequence_data(df)

    print("\n📦 Testing LSTM:")
    lstm_model = load_model("models/lstm_model.keras")
    lstm_pred = lstm_model.predict(X_seq[-1:])[0][0]
    print(f"LSTM prediction: {lstm_pred:.2f} Wh")

    print("\n📦 Testing TCN:")
    tcn_model = load_model("models/tcn_model.keras")
    tcn_pred = tcn_model.predict(X_seq[-1:])[0][0]
    print(f"TCN prediction: {tcn_pred:.2f} Wh")

# Main
if __name__ == "__main__":
    df = load_data()
    predict_tree_models(df)
    predict_sequence_models(df)
