import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import config as cfg
from supervised_learning.models.staf_net import TemporalAttentiveFusionNet as STAF


# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def _read_clean_csv(path):
    """
    Reads a CSV, cleans string artifacts, and handles missing values.
    Returns a dataframe with DateTime index.
    """
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Returning Mock Data.")
        dates = pd.date_range(start=cfg.START_DATE, end=cfg.END_DATE, freq=cfg.RESOLUTION)
        return pd.DataFrame(np.random.uniform(10, 100, size=(len(dates), 1)), index=dates)

    df = pd.read_csv(path, index_col=0)
    
    # Remove quotes and whitespace from string columns
    df = df.apply(lambda c: c.astype(str).str.replace('"', '').str.strip() if c.dtype == 'object' else c)
    
    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Handle index
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        print(f"Warning: Could not convert index to datetime for {path}")

    # Fill gaps
    df = df.ffill().bfill()
    return df

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_and_prepare_data():
    """
    Loads your CSVs using paths from config.
    """
    print("Loading data...")
    
    # 2a. Load Battery Demand
    if os.path.exists(cfg.BATTERY_FILE):
        df_batt = pd.read_csv(cfg.BATTERY_FILE, index_col=0)
        df_batt.index = pd.to_datetime(df_batt.index)
        df_batt = df_batt.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    else:
        print(f"Battery file not found: {cfg.BATTERY_FILE}. Generating mock.")
        dates = pd.date_range(start=cfg.START_DATE, end=cfg.END_DATE, freq=cfg.RESOLUTION)
        df_batt = pd.DataFrame({'battery_demand': np.random.uniform(0, 10, size=len(dates))}, index=dates)
    
    # 2b. Load Environmental Data
    b_demand = _read_clean_csv(f"{cfg.TS_DIR}/building_data.csv")
    price = _read_clean_csv(f"{cfg.TS_DIR}/electricitycostG2B_data.csv")
    rad = _read_clean_csv(f"{cfg.TS_DIR}/radiation_data.csv")
    temp = _read_clean_csv(f"{cfg.TS_DIR}/temperature_data.csv")
    
    return b_demand, rad, temp, price, df_batt

# ==========================================
# 3. EXTRACTION LOGIC
# ==========================================
def get_test_features(test_dates, b_demand, rad, temp, price, battery_demand, window = 0):
    """
    Extracts a 24-hour data slice based on the first date in test_dates, 
    starting with an hourly offset defined by 'window'.
    
    Args:
        test_dates (list): A list of date strings (only the first is used).
        ... (DataFrames): Timeseries dataframes.
        window (int): Hourly offset (0-23 hours) for the start of the 24h slice.
        
    Returns:
        numpy.ndarray: A single flattened array of all 5 features over 24 steps (length 120), 
                       or None on error.
    """
    start_date_str = test_dates[0]
    
    # Define the precise start timestamp with the hourly window offset
    ts_start_base = pd.to_datetime(start_date_str).replace(hour=0, minute=0, second=0)
    ts_start = ts_start_base + pd.Timedelta(hours=window)
    
    # Define the precise end timestamp (23 hours after ts_start, inclusive)
    ts_end = ts_start + pd.Timedelta(hours=23) 
    
    print(f"Extracting Data from: {ts_start} to {ts_end}")
    
    # --- Data Slicing and Validation ---
    
    # Validate index range availability
    # We check if the timestamps exist in the index, which is safer than checking only the base date.
    if ts_start not in b_demand.index or ts_end not in b_demand.index:
        print(f"Error: Range {ts_start} to {ts_end} not found in data index.")
        return None
    
    # Slice all Dataframes using the calculated windowed range
    # Note: .loc works on the DatetimeIndex for all inputs, including battery_demand
    slice_demand = b_demand.loc[ts_start:ts_end].iloc[:, 0].values.flatten()
    slice_rad = rad.loc[ts_start:ts_end].iloc[:, 0].values.flatten()
    slice_temp = temp.loc[ts_start:ts_end].iloc[:, 0].values.flatten()
    slice_price = price.loc[ts_start:ts_end].iloc[:, 0].values.flatten()
    slice_batt = battery_demand.loc[ts_start:ts_end].iloc[:, 0].values.flatten()
    
    if len(slice_demand) != 24:
        print(f"WARNING: Expected 24 time steps, got {len(slice_demand)}. Check data gaps or resolution.")
        # Decide if you want to proceed with non-24 length data, or return None.
        # For model input, returning None might be safer if the sequence length is strict.
        # return None

    # --- Feature Construction ---
    
    # Concatenate all 5 feature arrays into a single, flat 1D NumPy array (length 120)
    env_feats_flat = np.concatenate([
        slice_demand,
        slice_rad,
        slice_temp,
        slice_price,
        slice_batt
    ])
    
    if len(env_feats_flat) != 120:
        print(f"WARNING: Expected 120 features (24*5), got {len(env_feats_flat)}.")

    # Return the single flat array.
    return env_feats_flat

# ==========================================
# 4. MODEL UTILS
# ==========================================
def load_model(model_path="bestmodel.pth"):
    """
    Loads the STAF model from the .pth file.
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Using initialized placeholder.")
        return STAF() # Return un-trained placeholder for testing flow
    
    try:
        dims= {'static': 4, 'series_count': 5, 'seq_len': 24, 'vehicle': 76, 'emb_vocab': 8023}
        # Assuming the .pth contains the state_dict (weights)
        # You need to instantiate the architecture first
        model = STAF(
            num_static=dims["static"],
            num_series=dims["series_count"],
            sequence_length=dims["seq_len"],
            vehicle_input_dim=dims["vehicle"],
            
            # Hyperparameters from Config
            num_embeddings=dims["emb_vocab"],
            embedding_dim=cfg.EMBEDDING_DIM,
            n_heads=cfg.N_HEADS,
            fc_hidden_dim1=cfg.HIDDEN_DIM_1,
            fc_hidden_dim2=cfg.HIDDEN_DIM_2,
            dropout=cfg.DROPOUT
        ).to(cfg.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval() # Set to evaluation mode
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Attempting to load as full model object...")
        try:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            model.eval()
            return model
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return None

def prepare_model_inputs(env_feats):
    """
    Converts list of numpy arrays to a single PyTorch tensor.
    Input: List of 5 arrays, each shape (48,)
    Output: Tensor of shape (1, 48, 5) -> (Batch, Seq, Features)
    """
    # 1. Stack features column-wise: Result shape (48, 5)
    data_array = np.stack(env_feats, axis=1)
    
    # 2. Convert to Tensor
    tensor = torch.tensor(data_array, dtype=torch.float32)
    
    # 3. Add Batch Dimension: (1, 48, 5)
    tensor = tensor.unsqueeze(0)
    
    return tensor

def run_inference(model, input_tensor):
    """
    Runs the model on the input tensor and returns the result.
    """
    with torch.no_grad():
        output = model(input_tensor)
        
    # Convert back to python/numpy scalar or array
    if output.numel() == 1:
        return output.item()
    else:
        return output.numpy()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def test_pipeline(case_id, tolerance, run_name):
    # 1. Load Data
    b_demand_df, rad_df, temp_df, price_df, batt_df = load_and_prepare_data()
    model_path = os.path.join("data/training_results", run_name, "checkpoints", "best_model.pth")
    # 2. Load Model
    model = load_model(model_path)
    if model is None:
        print("Critial: Could not load model.")
        return

    # 3. Define Test Cases
    cases = [
        ("Test Case 1", cfg.TEST_DATE1),
        ("Test Case 2", cfg.TEST_DATE2)
    ]

    # 4. Loop through cases
    for case_name, date_range in cases:
        print(f"\n--- Processing {case_name} ---")
        
        # A. Extract Features
        feats = get_test_features(
            date_range, 
            b_demand_df, rad_df, temp_df, price_df, batt_df
        )
        
        if feats is None:
            continue

        # B. Prepare for Model
        input_tensor = prepare_model_inputs(feats)
        print(f"Input Tensor Shape: {input_tensor.shape}") # Should be (1, 48, 5)

        # C. Run Inference
        result = run_inference(model, input_tensor)
        print(f"Model Output Value: {result}")

if __name__ == "__main__":
    test_pipeline()