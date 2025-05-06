import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap
from lime.lime_tabular import LimeTabularExplainer
from lightgbm import LGBMRegressor
import joblib
import os


# Load the dataset
def load_data(path="data/energydata_complete.csv"):
    df = pd.read_csv(path)
    print("Data loaded successfully.")
    return df

# Preprocess the dataset
def preprocess_data(df):
    # Let pandas infer the date format automatically
    df["date"] = pd.to_datetime(df["date"])

    # Extract hour, day of the week, and weekend flag (0 or 1)
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)  # 0 or 1

    # Drop columns not needed
    df = df.drop(columns=["date", "rv1", "rv2"])

    # Convert remaining features (excluding Appliances) to float64
    for col in df.columns:
        if col != "Appliances":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.astype({col: 'float64' for col in df.columns if col != "Appliances"})

    print("Date parsed correctly and columns extracted:")
    print(df[["hour", "day_of_week", "is_weekend"]].head())

    return df


# Train an XGBoost model
def train_xgboost(df):
    X = df.drop(columns=["Appliances"])
    y = df["Appliances"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n📊 XGBoost Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    return model, X_train

# Generate SHAP summary plot
def explain_with_shap(model, X_train, model_name="model"):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    print(f"Generating SHAP summary plot for {model_name}...")
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(f"shap_summary_{model_name}.png")
    plt.clf()  # Clear the current figure
    print(f"Saved SHAP summary plot as shap_summary_{model_name}.png")


# Generate SHAP force plot for a single prediction
def explain_single_prediction(model, X, index=0, model_name="model"):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    print(f"Generating SHAP force plot for sample index {index} using {model_name}...")
    shap.plots.force(shap_values[index], matplotlib=True, show=False)
    plt.savefig(f"shap_force_{model_name}_{index}.png")
    plt.clf()
    print(f"Saved SHAP force plot as shap_force_{model_name}_{index}.png")



def explain_with_lime(model, X_train, index=0, model_name="model"):
    print(f"\nGenerating LIME explanation for sample index {index} using {model_name}...")

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        mode="regression"
    )

    instance = X_train.iloc[index].values
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict
    )

    # Save the explanation as HTML
    explanation.save_to_file(f"lime_explanation_{model_name}_{index}.html")
    print(f"LIME explanation saved to lime_explanation_{model_name}_{index}.html")



def train_lightgbm(df):
    X = df.drop(columns=["Appliances"])
    y = df["Appliances"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n📊 LightGBM Results:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return model, X_train



if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)

    os.makedirs("models", exist_ok=True)

    # Train XGBoost
    model_xgb, X_train_xgb = train_xgboost(df)
    explain_with_shap(model_xgb, X_train_xgb, model_name="xgboost")
    explain_single_prediction(model_xgb, X_train_xgb, index=0, model_name="xgboost")
    explain_with_lime(model_xgb, X_train_xgb, index=0, model_name="xgboost")
    joblib.dump(model_xgb, "models/xgboost_model.pkl")  # Save XGBoost

    # Train LightGBM
    model_lgb, X_train_lgb = train_lightgbm(df)
    explain_with_shap(model_lgb, X_train_lgb, model_name="lightgbm")
    explain_single_prediction(model_lgb, X_train_lgb, index=0, model_name="lightgbm")
    explain_with_lime(model_lgb, X_train_lgb, index=0, model_name="lightgbm")
    joblib.dump(model_lgb, "models/lightgbm_model.pkl")  # Save LightGBM


