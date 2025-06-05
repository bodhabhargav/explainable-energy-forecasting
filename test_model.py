import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess dataset
def load_data(path="data/energydata_complete.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df.drop(columns=["date", "rv1", "rv2"], inplace=True)
    df = df.astype({col: 'float64' for col in df.columns if col != "Appliances"})
    return df

# Prepare sequential data for LSTM and TCN
def prepare_sequence_data(df, target_column="Appliances", sequence_length=24):
    df_seq = df.drop(columns=[target_column])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_seq)
    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i + sequence_length])
        y.append(df.iloc[i + sequence_length][target_column])
    return np.array(X), np.array(y)

# Evaluate all models and plot predictions
def evaluate_models(df, N=100):
    print("✅ Evaluating models on the last {} samples...\n".format(N))

    # Columns used for tree-based models
    feature_cols = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4',
                    'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8',
                    'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
                    'Visibility', 'Tdewpoint', 'hour', 'day_of_week', 'is_weekend',
                    'hour_sin', 'hour_cos']

    X_tree = df[feature_cols]
    y_true = df["Appliances"].values[-N:]

    # Load models
    xgb_model = joblib.load("models/xgboost_model.pkl")
    lgb_model = joblib.load("models/lightgbm_model.pkl")
    lstm_model = load_model("models/lstm_model.keras", compile=False)
    tcn_model = load_model("models/tcn_model.keras", compile=False)

    # Predict tree-based
    xgb_preds = []
    lgb_preds = []
    for i in range(len(df) - N, len(df)):
        xgb_pred = xgb_model.predict(X_tree.iloc[i:i+1])[0]
        lgb_pred = lgb_model.predict(X_tree.iloc[i:i+1])[0]
        xgb_preds.append(xgb_pred)
        lgb_preds.append(lgb_pred)

    print("📦 XGBoost sample predictions:", np.round(xgb_preds[:5], 2))
    print("📦 LightGBM sample predictions:", np.round(lgb_preds[:5], 2))

    # Predict sequence-based
    X_seq, y_seq = prepare_sequence_data(df)
    lstm_preds = []
    tcn_preds = []
    for i in range(len(X_seq) - N, len(X_seq)):
        lstm_pred = lstm_model.predict(X_seq[i:i+1], verbose=0)[0][0]
        tcn_pred = tcn_model.predict(X_seq[i:i+1], verbose=0)[0][0]
        lstm_preds.append(lstm_pred)
        tcn_preds.append(tcn_pred)

    print("📦 LSTM sample predictions:", np.round(lstm_preds[:5], 2))
    print("📦 TCN sample predictions:", np.round(tcn_preds[:5], 2))
    print()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True", linewidth=2)
    plt.plot(xgb_preds, label="XGBoost")
    plt.plot(lgb_preds, label="LightGBM")
    plt.plot(lstm_preds, label="LSTM")
    plt.plot(tcn_preds, label="TCN")
    plt.title("Energy Appliance Predictions (Last {} Samples)".format(N))
    plt.xlabel("Time Step")
    plt.ylabel("Appliances Energy Consumption (Wh)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("model_comparison_plot.png")
    plt.show()
    print("📊 Plot saved as: model_comparison_plot.png")

# Main execution
if __name__ == "__main__":
    df = load_data()
    evaluate_models(df, N=100)
