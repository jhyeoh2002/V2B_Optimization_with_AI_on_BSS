import pandas as pd
import numpy as np
import os
import json
import torch
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from supervised_learning.models.staf_net import TemporalAttentiveFusionNet as STAF
from util.case_dir import case_dir

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def _calculate_priority(soc_current, dep_soc, dep_time, current_hour):
    """
    Calculates charging priority based on remaining time and required charge.
    
    Returns:
        np.array: Priority scores.
    """
    denom = (dep_time - current_hour)
    # Avoid division by zero
    denom[denom == 0] = np.nan
    
    priority = (dep_soc - soc_current) / (denom + 1e-6)
    return priority

def _read_clean_csv(path):
    """
    Reads a CSV, cleans string artifacts, and handles missing values.
    Returns a dataframe with DateTime index.
    """
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None

    df = pd.read_csv(path, index_col=0)
    
    # Clean artifacts
    df = df.apply(lambda c: c.astype(str).str.replace('"', '').str.strip() if c.dtype == 'object' else c)
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Handle index
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        print(f"[WARN] Could not convert index to datetime for {path}")

    # Fill gaps
    df = df.ffill().bfill()
    return df

def load_and_prepare_data(case_id):
    """
    Loads environmental and battery demand data.
    """
    print(f"\t[INFO] Loading validation data for Case {case_id}...")
    
    # Paths (adjusting to your project structure)
    ts_dir = "./data/timeseries"
    
    # Load separate series
    b_demand = _read_clean_csv(f"{ts_dir}/building_data.csv")
    price = _read_clean_csv(f"{ts_dir}/electricitycostG2B_data.csv")
    rad = _read_clean_csv(f"{ts_dir}/radiation_data.csv")
    temp = _read_clean_csv(f"{ts_dir}/temperature_data.csv")
    batt_df = _read_clean_csv(f"data/battery_demand/resample_full.csv")
    

    return b_demand, rad, temp, price, batt_df

def _generate_cyclic_features(index):
    """
    Creates continuous cyclic representations (Sin/Cos) for Hour and Day.
    
    Args:
        index (pd.DatetimeIndex): The time index of the data.
        
    Returns:
        pd.DataFrame: DataFrame containing sin/cos features.
    """
    df = pd.DataFrame(index=index)
    df["sin_hour"] = np.sin(2 * np.pi * index.hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * index.hour / 24)
    df["sin_day"] = np.sin(2 * np.pi * index.dayofweek / 7)
    df["cos_day"] = np.cos(2 * np.pi * index.dayofweek / 7)
    return df.round(5)

# ==========================================
# 2. FEATURE CONSTRUCTION (The Critical Fix)
# ==========================================
def get_full_test_vector(i, iter, b_demand, rad, temp, price, battery_demand, feature_info):
    """
    Constructs the EXACT input vector the model expects:
    [Static (4) | Flattened Series (120) | Vehicle Priorities (76)]
    """
    # 1. Parse Time
    demand_dir = cfg.BATTERYDEMAND_DIR
    case_demand_dir = case_dir(demand_dir, 0)
    
    battery_demand = np.load(f"{case_demand_dir}/battery_demand.npy")
    battery_details = np.load(f"{case_demand_dir}/battery_details.npy") # [arr_soc, dep_soc, arr_time, dep_time]
    battery_schedule = np.load(f"{case_demand_dir}/battery_availability.npy")
    hour_now = 23+iter
    idx =int( battery_demand[i,0])+iter
    battery_demand_i = battery_demand[i,1:]
    
    sched_row = battery_schedule[i]
    sched_row = sched_row[~np.isnan(sched_row).any(axis=1)]
    static_df = _generate_cyclic_features(b_demand.index) 

    battery_demand_feat = battery_demand_i[hour_now-23:hour_now+1]
       
    series_flat = np.concatenate([
        b_demand.iloc[idx:idx + 24, 0].values,
        rad.iloc[idx:idx + 24, 0].values,
        temp.iloc[idx:idx + 24, 0].values,
        price.iloc[idx:idx + 24, 0].values,
        battery_demand_feat
    ])
    
    expected_series_len = 5 * 24 # 120
    if len(series_flat) != expected_series_len:
        print(f"\t[ERR] Shape Mismatch. Expected {expected_series_len} series points, got {len(series_flat)}")
        return None

    # 4. Generate Vehicle Features (76)
        # Slice current timestep data
    SOC = np.load(f"data/optimization_results/case0_test/optimization/SOC.npy")
    soc_row = SOC[i].T
    soc_row = soc_row[~np.isnan(soc_row).any(axis=1)]
    
    
    avail_flags = sched_row[:, hour_now]
    active_mask = avail_flags > 0
    curr_soc = soc_row[active_mask, hour_now]
    departure_soc = battery_details[1]
    departure_times = battery_details[3]
    # Get matching departure info
    curr_dep_soc = departure_soc[i][~np.isnan(departure_soc[i])][active_mask]
    curr_dep_time = departure_times[i][~np.isnan(departure_times[i])][active_mask]
    
    # # Update detected vehicles for column naming later
    # if len(curr_soc) > n_veh_detected:
    #     n_veh_detected = len(curr_soc)

    # Calculate Priority
    priority = _calculate_priority(curr_soc, curr_dep_soc, curr_dep_time, hour_now)
    
    n_veh = sum(len(x) for x in feature_info['battery_blocks'])
    vehicle_feats = np.zeros(n_veh) 

    # 5. Concatenate All -> [Static, Series, Vehicles]
    full_vector = np.concatenate([static_df.iloc[hour_now - 1, :].tolist(), series_flat, vehicle_feats])
    
    # Return as Tensor [1, Total_Dim]
    return torch.tensor(full_vector, dtype=torch.float32).unsqueeze(0)

# ==========================================
# 3. MODEL LOADING
# ==========================================
def load_trained_model(run_name):
    """
    Loads model using the specific JSON config from training to ensure shapes match.
    """
    # Construct paths
    base_path = os.path.join(cfg.TRAIN_RESULTS_DIR, run_name)
    json_path = os.path.join(base_path, f"feature_info_{run_name}.json")
    model_path = os.path.join(base_path, "checkpoints", "best_model.pth")
    
    if not os.path.exists(json_path):
        print(f"[ERR] Feature info not found: {json_path}")
        return None, None
    if not os.path.exists(model_path):
        print(f"[ERR] Model checkpoint not found: {model_path}")
        return None, None

    # A. Load Config
    with open(json_path, 'r') as f:
        feat_info = json.load(f)
        
    dims = {
        "static": len(feat_info["static_cols"]),
        "series_count": len(feat_info["series_blocks"]),
        "seq_len": len(feat_info["series_blocks"][0]),
        "vehicle": sum(len(block) for block in feat_info["battery_blocks"]),
        # Use get() in case key is missing (fallback to config default if needed)
        "emb_vocab": feat_info.get("num_embeddings", cfg.NUM_EMBEDDINGS)
    }
    
    print(f"\t[INFO] Initializing Model from {run_name} (Vocab: {dims['emb_vocab']})")

    # B. Init Model Architecture
    model = STAF(
        num_static=dims["static"],
        num_series=dims["series_count"],
        sequence_length=dims["seq_len"],
        vehicle_input_dim=dims["vehicle"],
        num_embeddings=dims["emb_vocab"], 
        embedding_dim=cfg.EMBEDDING_DIM,
        n_heads=cfg.N_HEADS,
        fc_hidden_dim1=cfg.HIDDEN_DIM_1,
        fc_hidden_dim2=cfg.HIDDEN_DIM_2,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)

    # C. Load Weights
    # map_location ensures it loads on CPU if CUDA is not available
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE, weights_only=True))
    model.eval()
    
    return model, feat_info

# ==========================================
# 4. MAIN PIPELINE WRAPPER
# ==========================================
def test_pipeline(case_id,  run_name):
    print(f"\n{'='*40}")
    print(f"🔬 RUNNING INFERENCE TEST: {run_name}")
    print(f"{'='*40}")

    # 1. Load Model & Meta-data
    model, feat_info = load_trained_model(run_name)
    if model is None:
        print("[FAIL] Could not load model. Aborting test.")
        return

    # 2. Load Data
    b_demand, rad, temp, price, batt = load_and_prepare_data(case_id)
    if batt is None:
        print("[FAIL] Battery data missing. Aborting.")
        return

    # 3. Define Test Dates (You can pull these from config if preferred)
    # Using a hardcoded list for demonstration, or cfg.TEST_DATES if you have it
    for i in range(2):
        
        for iter in range(24):
            # A. Prepare Input
            input_tensor = get_full_test_vector(
                i, iter, b_demand, rad, temp, price, batt, feat_info
            )
            
            if input_tensor is None:
                continue

            # B. Run Model
            input_tensor = input_tensor.to(cfg.DEVICE)
            with torch.no_grad():
                prediction = model(input_tensor)
                
            # C. Output Result
            val = prediction.item()
            print(f"   📅 {i} -> Pred: {val:.4f} kWh")

    print("\n✅ Test Pipeline Complete.")