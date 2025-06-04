import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

try:
    from tcn import TCN
except ImportError:
    print("keras-tcn not installed. TCNModel will not be available.")
    TCN = None

# --- XGBoostModel ---
class XGBoostModel:
    def __init__(self):
        from xgboost import XGBRegressor
        self.model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# --- LightGBMModel ---
class LightGBMModel:
    def __init__(self):
        self.model = LGBMRegressor(n_estimators=100, learning_rate=0.1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# --- LSTMModel ---
class LSTMModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=False),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y):
        self.model.fit(X, y, epochs=32, batch_size=32,
                       validation_split=0.2,
                       callbacks=[EarlyStopping(patience=5)],
                       verbose=0)

    def predict(self, X):
        return self.model.predict(X).flatten()

# --- TCNModel ---
class TCNModel:
    def __init__(self, input_shape):
        if TCN is None:
            raise ImportError("keras-tcn must be installed to use TCNModel")
        self.model = Sequential([
            Input(shape=input_shape),
            TCN(64),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, y):
        self.model.fit(X, y, epochs=32, batch_size=32,
                       validation_split=0.2,
                       callbacks=[EarlyStopping(patience=5)],
                       verbose=0)

    def predict(self, X):
        return self.model.predict(X).flatten()

# --- Utility function ---
def create_sequences(X_data, y_data, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - timesteps):
        X_seq.append(X_data[i:i + timesteps])
        y_seq.append(y_data[i + timesteps])
    return np.array(X_seq), np.array(y_seq)
