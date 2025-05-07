🔍 Explainable Energy Forecasting

This project uses machine learning models (XGBoost and LightGBM) to predict household energy consumption and provides explainability using SHAP and LIME techniques.

📂 Dataset

Source: UCI Machine Learning Repository

File: energydata_complete.csv

Description: Contains energy usage data (in Wh) from a smart home over a 6-week period, including environmental and weather-related features.

🛠️ Preprocessing

Parses the date column to extract:

hour (0–23)

day_of_week (0=Monday, 6=Sunday)

is_weekend (0 or 1)

Drops irrelevant features: rv1, rv2, and the original date.

Ensures all features (excluding target) are converted to float64.

🧠 Models Used:

XGBoost Regressor

LightGBM Regressor

Both models are trained to predict the Appliances energy consumption column.

📈 Evaluation Metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

These metrics are printed after training.

📊 Explainability:

SHAP (SHapley Additive exPlanations):

Global interpretation using summary plots

Local interpretation using force plots

LIME (Local Interpretable Model-agnostic Explanations):

Explains a specific instance and saves it as an HTML file

📁 Output Files:

shap_summary_xgboost.png, shap_summary_lightgbm.png

shap_force_xgboost_0.png, shap_force_lightgbm_0.png

lime_explanation_xgboost_0.html, lime_explanation_lightgbm_0.html

🚀 Running the Project:

# Install dependencies
pip install -r requirements.txt

# Run the main training + explainability script
python main.py

📊 Streamlit App:

You can also run a visual interface using:

bash

streamlit run streamlit_app.py
(Make sure you have the streamlit package downloaded.)

📌 Project Goals:

Build accurate energy consumption forecasting models

Enable transparency with model predictions

Support interpretability using both model-specific and model-agnostic methods

