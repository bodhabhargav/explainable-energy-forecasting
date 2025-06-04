# Integrated SHAP & LIME into the full pipeline
import os
import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load

from models import XGBoostModel, LightGBMModel, LSTMModel, TCNModel, create_sequences

import shap
import lime
import lime.lime_tabular
from keras.models import load_model as keras_load

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainingLogger")

# --- Configuration ---
DATA_FILE = 'data/energydata_processed.csv'
TARGET = 'Appliances'
OUTPUT_DIR = 'outputs'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
EXPLAIN_DIR = os.path.join(OUTPUT_DIR, 'explainability')
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'model_results.json')

TIMESTEPS = 24  # 4 hours if data is at 10-min intervals
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPLAIN_DIR, exist_ok=True)

# --- Load and prepare data ---
df = pd.read_csv(DATA_FILE)
y = df[TARGET]
X = df.drop(columns=[TARGET])
feature_names = X.columns
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost ---
logger.info("Training XGBoost...")
xgb = XGBoostModel()
xgb.fit(X_train_2d, y_train_2d)
y_pred_xgb = xgb.predict(X_test_2d)
rmse_xgb = np.sqrt(mean_squared_error(y_test_2d, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test_2d, y_pred_xgb)
logger.info(f"XGBoost: RMSE = {rmse_xgb:.3f}, MAE = {mae_xgb:.3f}")
dump(xgb.model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))

# SHAP + LIME for XGBoost
explainer_xgb = shap.Explainer(xgb.model)
shap_values_xgb = explainer_xgb(X_test_2d)
shap.summary_plot(shap_values_xgb, X_test_2d, show=False)
plt.savefig(os.path.join(EXPLAIN_DIR, "xgb_shap_summary.png"))
plt.clf()
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_2d), feature_names=list(feature_names), mode="regression")
exp = lime_explainer.explain_instance(X_test_2d.iloc[0], xgb.model.predict)
exp.save_to_file(os.path.join(EXPLAIN_DIR, "xgb_lime_explanation.html"))

# --- Train LightGBM ---
logger.info("Training LightGBM...")
lgbm = LightGBMModel()
lgbm.fit(X_train_2d, y_train_2d)
y_pred_lgbm = lgbm.predict(X_test_2d)
rmse_lgbm  = np.sqrt(mean_squared_error(y_test_2d, y_pred_lgbm))
mae_lgbm = mean_absolute_error(y_test_2d, y_pred_lgbm)
logger.info(f"LightGBM: RMSE = {rmse_lgbm:.3f}, MAE = {mae_lgbm:.3f}")
dump(lgbm.model, os.path.join(MODELS_DIR, "lightgbm_model.joblib"))

# SHAP + LIME for LightGBM
explainer_lgbm = shap.Explainer(lgbm.model)
shap_values_lgbm = explainer_lgbm(X_test_2d)
shap.summary_plot(shap_values_lgbm, X_test_2d, show=False)
plt.savefig(os.path.join(EXPLAIN_DIR, "lgbm_shap_summary.png"))
plt.clf()
exp = lime_explainer.explain_instance(X_test_2d.iloc[0], lgbm.model.predict)
exp.save_to_file(os.path.join(EXPLAIN_DIR, "lgbm_lime_explanation.html"))

# --- Sequence Prep for DL Models ---
X_seq, y_seq = create_sequences(X_train_2d.values, y_train_2d.values, TIMESTEPS)
X_test_seq, y_test_seq = create_sequences(X_test_2d.values, y_test_2d.values, TIMESTEPS)

# --- Train LSTM ---
logger.info("Training LSTM...")
lstm = LSTMModel(input_shape=(TIMESTEPS, X_train_2d.shape[1]))
lstm.fit(X_seq, y_seq)
y_pred_lstm = lstm.predict(X_test_seq)
rmse_lstm  = np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))
mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
logger.info(f"LSTM: RMSE = {rmse_lstm:.3f}, MAE = {mae_lstm:.3f}")
lstm.model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))

# LIME for LSTM
X_test_flat = X_test_seq.reshape((X_test_seq.shape[0], -1))
lime_explainer_lstm = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_test_flat,
    feature_names=[f"f{i}" for i in range(X_test_flat.shape[1])],
    mode="regression")
exp = lime_explainer_lstm.explain_instance(X_test_flat[0], lambda x: lstm.model.predict(x.reshape((-1, TIMESTEPS, X.shape[1]))).flatten())
exp.save_to_file(os.path.join(EXPLAIN_DIR, "lstm_lime_explanation.html"))

# --- Train TCN ---
logger.info("Training TCN...")
tcn = TCNModel(input_shape=(TIMESTEPS, X_train_2d.shape[1]))
tcn.fit(X_seq, y_seq)
y_pred_tcn = tcn.predict(X_test_seq)
rmse_tcn   = np.sqrt(mean_squared_error(y_test_seq, y_pred_tcn))
mae_tcn = mean_absolute_error(y_test_seq, y_pred_tcn)
logger.info(f"TCN: RMSE = {rmse_tcn:.3f}, MAE = {mae_tcn:.3f}")
tcn.model.save(os.path.join(MODELS_DIR, "tcn_model.keras"))

# LIME for TCN
lime_explainer_tcn = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_test_flat,
    feature_names=[f"f{i}" for i in range(X_test_flat.shape[1])],
    mode="regression")
exp = lime_explainer_tcn.explain_instance(X_test_flat[0], lambda x: tcn.model.predict(x.reshape((-1, TIMESTEPS, X.shape[1]))).flatten())
exp.save_to_file(os.path.join(EXPLAIN_DIR, "tcn_lime_explanation.html"))

# --- Save Metrics ---
results = {
    "xgboost": {"rmse": rmse_xgb, "mae": mae_xgb},
    "lightgbm": {"rmse": rmse_lgbm, "mae": mae_lgbm},
    "lstm": {"rmse": rmse_lstm, "mae": mae_lstm},
    "tcn": {"rmse": rmse_tcn, "mae": mae_tcn},
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=4)

logger.info("All model results and explainability outputs saved.")

