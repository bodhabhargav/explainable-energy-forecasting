# data_preprocessing.py

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose # For seasonal decomposition
import os
import logging

# --- Configuration ---
DATA_FILE = 'data/energydata_complete.csv'
OUTPUT_DIR = 'outputs'
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'EDA')
LOG_FILE = os.path.join(OUTPUT_DIR, 'pre_processing.log')
OUTPUT_PROCESSED_FILE = os.path.join('data', 'energydata_processed.csv')
LAG_COLUMNS = ['Appliances', 'T_out', 'RH_out'] # Columns to create lag features for
N_LAGS = 0 # Number of lags to create

# --- Setup Output Directories and Logging ---
def setup_environment():
    """Creates output directories if they don't exist and configures logging."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'), # Overwrite log file each run
            logging.StreamHandler() # Also print to console
        ]
    )
    logging.info("Environment setup complete. Logging configured.")

# Call setup early
setup_environment()

# --- 1. Load Data ---
def load_data(file_path):
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"Data file '{file_path}' not found.")
        return None
    logging.info(f"Loading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

# --- 2. Summarize Data ---
def summarize_data(df):
    """Logs a summary of the DataFrame."""
    if df is None:
        return
    logging.info("--- Data Summary ---")
    logging.info(f"First 5 rows:\n{df.head()}")
    
    # Capture df.info() output
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    logging.info(f"DataFrame Info:\n{info_str}")
    
    logging.info(f"Descriptive Statistics:\n{df.describe()}")
    logging.info(f"Missing Values per column:\n{df.isnull().sum()}")

# --- 3. Visualize Data ---
def visualize_data_exploratory(df):
    """Generates and saves exploratory visualizations."""
    if df is None:
        logging.warning("DataFrame is None, skipping exploratory visualizations.")
        return
    logging.info("--- Generating Exploratory Visualizations ---")

    df_vis = df.copy()
    if 'date' in df_vis.columns:
        try:
            df_vis['date'] = pd.to_datetime(df_vis['date'])
            df_vis.set_index('date', inplace=True, drop=False) # Keep date column too if needed later
        except Exception as e:
            logging.error(f"Error converting 'date' column for visualization: {e}")
            # Proceed without date index if conversion fails, some plots might not work
    
    # Plot 1: Time series of 'Appliances' energy consumption
    try:
        plt.figure(figsize=(15, 6))
        plt.plot(df_vis.index, df_vis['Appliances'], label='Appliances Energy (Wh)', color='royalblue')
        plt.title('Appliances Energy Consumption Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Energy (Wh)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, '01_appliances_time_series.png'))
        plt.close()
        logging.info("Saved: 01_appliances_time_series.png")
    except Exception as e:
        logging.error(f"Error generating appliances time series plot: {e}")

    # Plot 2: Distribution of 'Appliances' energy consumption
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_vis['Appliances'], kde=True, color='forestgreen')
        plt.title('Distribution of Appliances Energy Consumption', fontsize=16)
        plt.xlabel('Energy (Wh)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, '02_appliances_distribution.png'))
        plt.close()
        logging.info("Saved: 02_appliances_distribution.png")
    except Exception as e:
        logging.error(f"Error generating appliances distribution plot: {e}")

    # Plot 3: Correlation Heatmap (numerical features only)
    try:
        plt.figure(figsize=(20, 18)) # Increased size
        numerical_df = df_vis.select_dtypes(include=np.number)
        # Drop rv1 and rv2 if they exist, as they are random and not informative
        if 'rv1' in numerical_df.columns: numerical_df = numerical_df.drop(columns=['rv1'])
        if 'rv2' in numerical_df.columns: numerical_df = numerical_df.drop(columns=['rv2'])
        
        corr_matrix = numerical_df.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, '03_correlation_heatmap.png'))
        plt.close()
        logging.info("Saved: 03_correlation_heatmap.png")
    except Exception as e:
        logging.error(f"Error generating correlation heatmap: {e}")
        
    # Plot 4: Seasonal Decomposition of 'Appliances'
    # Requires data to be sampled at a consistent frequency and have a date index.
    # The data is 10-minutely. Period for daily seasonality: 6 samples/hr * 24 hrs = 144
    # Period for weekly seasonality: 144 samples/day * 7 days = 1008
    # Using daily seasonality for this example.
    if isinstance(df_vis.index, pd.DatetimeIndex) and len(df_vis) > 2 * 144 : # Ensure enough data for decomposition
        try:
            decomposition = seasonal_decompose(df_vis['Appliances'], model='additive', period=144, extrapolate_trend='freq')
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
            decomposition.observed.plot(ax=ax1, legend=False, color='royalblue')
            ax1.set_ylabel('Observed')
            decomposition.trend.plot(ax=ax2, legend=False, color='forestgreen')
            ax2.set_ylabel('Trend')
            decomposition.seasonal.plot(ax=ax3, legend=False, color='orangered')
            ax3.set_ylabel('Seasonal (Daily)')
            decomposition.resid.plot(ax=ax4, legend=False, color='purple')
            ax4.set_ylabel('Residual')
            plt.suptitle('Seasonal Decomposition of Appliances Energy Consumption (Daily Seasonality)', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plt.savefig(os.path.join(VISUALIZATION_DIR, '04_appliances_seasonal_decomposition.png'))
            plt.close()
            logging.info("Saved: 04_appliances_seasonal_decomposition.png")
        except Exception as e:
            logging.error(f"Error generating seasonal decomposition plot: {e}")
    else:
        logging.warning("Skipping seasonal decomposition: Date index not set, data too short, or other issue.")

    # Plot 5: Box plot of 'Appliances' vs. 'hour' (requires 'hour' feature)
    if 'date' in df.columns: # Use original df for feature availability before indexing
        temp_df_for_hour_plot = df.copy()
        temp_df_for_hour_plot['hour'] = pd.to_datetime(temp_df_for_hour_plot['date']).dt.hour
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='hour', y='Appliances', data=temp_df_for_hour_plot, palette='viridis')
            plt.title('Appliances Energy vs. Hour of Day', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=12)
            plt.ylabel('Energy (Wh)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_DIR, '05_appliances_vs_hour_boxplot.png'))
            plt.close()
            logging.info("Saved: 05_appliances_vs_hour_boxplot.png")
        except Exception as e:
            logging.error(f"Error generating appliances vs hour boxplot: {e}")

    # Plot 6: Box plot of 'Appliances' vs. 'day_of_week'
    if 'date' in df.columns:
        temp_df_for_dow_plot = df.copy()
        temp_df_for_dow_plot['day_of_week'] = pd.to_datetime(temp_df_for_dow_plot['date']).dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='day_of_week', y='Appliances', data=temp_df_for_dow_plot, order=day_order, palette='plasma')
            plt.title('Appliances Energy vs. Day of Week', fontsize=16)
            plt.xlabel('Day of Week', fontsize=12)
            plt.ylabel('Energy (Wh)', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_DIR, '06_appliances_vs_day_of_week_boxplot.png'))
            plt.close()
            logging.info("Saved: 06_appliances_vs_day_of_week_boxplot.png")
        except Exception as e:
            logging.error(f"Error generating appliances vs day_of_week boxplot: {e}")

    # Plot 7: Scatter plot of 'Appliances' vs. 'T_out'
    if 'T_out' in df_vis.columns:
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='T_out', y='Appliances', data=df_vis, alpha=0.5, color='orangered', s=10) # s for marker size
            plt.title('Appliances Energy vs. Outside Temperature (T_out)', fontsize=16)
            plt.xlabel('Outside Temperature (°C)', fontsize=12)
            plt.ylabel('Energy (Wh)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_DIR, '07_appliances_vs_T_out_scatter.png'))
            plt.close()
            logging.info("Saved: 07_appliances_vs_T_out_scatter.png")
        except Exception as e:
            logging.error(f"Error generating appliances vs T_out scatter plot: {e}")

    # Plot 8: Rolling Mean and Std Dev of 'Appliances'
    if isinstance(df_vis.index, pd.DatetimeIndex):
        try:
            rolling_window = 144 # 1 day rolling window (144 samples for 10-min data)
            plt.figure(figsize=(15, 6))
            plt.plot(df_vis.index, df_vis['Appliances'], label='Original Appliances', color='lightgrey', alpha=0.6)
            plt.plot(df_vis.index, df_vis['Appliances'].rolling(window=rolling_window).mean(), label=f'{rolling_window//144}-Day Rolling Mean', color='royalblue')
            plt.plot(df_vis.index, df_vis['Appliances'].rolling(window=rolling_window).std(), label=f'{rolling_window//144}-Day Rolling Std Dev', color='orangered', linestyle='--')
            plt.title('Appliances Energy with Rolling Mean & Standard Deviation', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Energy (Wh)', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALIZATION_DIR, '08_appliances_rolling_stats.png'))
            plt.close()
            logging.info("Saved: 08_appliances_rolling_stats.png")
        except Exception as e:
            logging.error(f"Error generating rolling statistics plot: {e}")

    logging.info("Exploratory visualizations generation attempt finished.")


# --- 4. Apply Preprocessing Steps ---
def preprocess_data(df):
    """Applies preprocessing steps as outlined in the research paper."""
    if df is None:
        logging.warning("DataFrame is None, skipping preprocessing.")
        return None
    logging.info("--- Preprocessing Data ---")
    processed_df = df.copy()

    # Drop rv1 and rv2
    if 'rv1' in processed_df.columns and 'rv2' in processed_df.columns:
        logging.info("Dropping 'rv1' and 'rv2' columns...")
        processed_df = processed_df.drop(columns=['rv1', 'rv2'])

    logging.info("Performing Time Feature Engineering...")
    if 'date' not in processed_df.columns:
        logging.error("'date' column not found for time feature engineering.")
        return None
        
    processed_df['date'] = pd.to_datetime(processed_df['date'])
    
    processed_df['hour'] = processed_df['date'].dt.hour
    processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
    processed_df['day_of_year'] = processed_df['date'].dt.dayofyear
    processed_df['month'] = processed_df['date'].dt.month
    processed_df['week_of_year'] = processed_df['date'].dt.isocalendar().week.astype(int) # Added week of year
    processed_df['weekend'] = processed_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    processed_df['hour_sin'] = np.sin(processed_df['hour'] * (2. * np.pi / 24))
    processed_df['hour_cos'] = np.cos(processed_df['hour'] * (2. * np.pi / 24))
    processed_df['day_of_week_sin'] = np.sin(processed_df['day_of_week'] * (2. * np.pi / 7))
    processed_df['day_of_week_cos'] = np.cos(processed_df['day_of_week'] * (2. * np.pi / 7))
    processed_df['month_sin'] = np.sin((processed_df['month'] -1) * (2. * np.pi / 12)) # month is 1-12
    processed_df['month_cos'] = np.cos((processed_df['month'] -1) * (2. * np.pi / 12))
    
    logging.info(f"Creating {N_LAGS} lag features for columns: {LAG_COLUMNS}...")
    for col in LAG_COLUMNS:
        if col not in processed_df.columns:
            logging.warning(f"Column '{col}' not found for lag feature creation. Skipping.")
            continue
        for i in range(1, N_LAGS + 1):
            processed_df[f'{col}_lag{i}'] = processed_df[col].shift(i)
            
    initial_rows = len(processed_df)
    processed_df = processed_df.dropna().reset_index(drop=True) # Reset index after dropping NaNs
    dropped_rows = initial_rows - len(processed_df)
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} rows due to NaN values from lag features.")
    else:
        logging.info("No rows dropped due to NaNs from lag features (or no lags created).")

    logging.info("Scaling numerical features...")
    features_to_scale = processed_df.select_dtypes(include=np.number).columns.tolist()
    
    # Exclude target and other non-scaleable identifiers if any
    # Original date components (like 'hour', 'month') can be scaled or left as is.
    # Cyclical features are already in [-1, 1] range, but scaling them usually doesn't hurt.
    # The main target 'Appliances' should not be scaled here if it's the Y variable.
    if 'Appliances' in features_to_scale:
        features_to_scale.remove('Appliances')
    
    # It's often better to not scale binary flags like 'weekend'
    if 'weekend' in features_to_scale:
        features_to_scale.remove('weekend')

    if not features_to_scale:
        logging.warning("No features identified or selected for scaling.")
    else:
        scaler = StandardScaler()
        # Note: In a full pipeline, fit scaler on TRAIN data only.
        processed_df[features_to_scale] = scaler.fit_transform(processed_df[features_to_scale])
        logging.info(f"Scaled features: {features_to_scale}")

    # Drop the original 'date' column as its information is now in engineered features
    # or it has been used as an index.
    if 'date' in processed_df.columns:
        logging.info("Dropping original 'date' column after feature engineering.")
        processed_df = processed_df.drop(columns=['date'])

    logging.info("Preprocessing completed.")
    return processed_df

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting data preprocessing script.")
    # 1. Load
    df_raw = load_data(DATA_FILE)

    if df_raw is not None:
        # 2. Summarize
        summarize_data(df_raw)
        
        # 3. Visualize (on raw data)
        visualize_data_exploratory(df_raw)
        
        # 4. Preprocess
        df_processed = preprocess_data(df_raw)
        
        if df_processed is not None:
            logging.info("--- Processed Data Summary ---")
            logging.info(f"First 5 rows of processed data:\n{df_processed.head()}")
            
            buffer = io.StringIO()
            df_processed.info(buf=buffer)
            info_str = buffer.getvalue()
            logging.info(f"Processed DataFrame Info:\n{info_str}")
            
            try:
                df_processed.to_csv(OUTPUT_PROCESSED_FILE, index=False)
                logging.info(f"Processed data saved to '{OUTPUT_PROCESSED_FILE}'")
            except Exception as e:
                logging.error(f"Error saving processed data: {e}")
        else:
            logging.error("Data preprocessing failed. Processed DataFrame is None.")
    else:
        logging.error("Data loading failed. Raw DataFrame is None.")
    
    logging.info("Data preprocessing script finished.")