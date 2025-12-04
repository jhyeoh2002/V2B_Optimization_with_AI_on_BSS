import os
import sys
import json
import numpy as np
import pandas as pd
from itertools import chain
from util.case_dir import case_dir

# Add project root to path if needed
sys.path.append(os.path.abspath(".."))

# ==============================================================================
# 1. HELPER FUNCTIONS (I/O & MATH)
# ==============================================================================

def _read_clean_csv(path):
    """
    Reads a CSV, cleans string artifacts, and handles missing values.
    
    Args:
        path (str): File path.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path, index_col=0)
    # Remove quotes and whitespace
    df = df.apply(lambda c: c.astype(str).str.replace('"', '').str.strip())
    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    # Fill gaps
    df = df.ffill().bfill()
    return df

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

def _analyze_quantization(all_series_list):
    """
    Helper to calculate suggested embedding size based on data range.
    """
    # Flatten all data to find global min/max/distribution
    all_values = list(chain(*[x.flatten() for x in all_series_list]))
    indices = np.round(np.array(all_values) * 1000) + 5000
    indices = np.clip(indices, 0, None).astype(int)
    num_unique = len(np.unique(indices))
    
    suggested_emb = int(num_unique * 1.2)
    print(f"\t\t[INFO] Unique quantized indices: {num_unique}")
    print(f"\t\t[INFO] Suggested num_embeddings: {suggested_emb}")
    return suggested_emb

# ==============================================================================
# 2. CORE LOGIC (WINDOWING)
# ==============================================================================

def create_sliding_windows(
    sequence_length, 
    battery_demand, 
    battery_schedule, 
    SOC, 
    V2B, 
    G2V, 
    P2V,
    departure_soc, 
    departure_times, 
    static_df, 
    env_data, 
    ground_truth
):
    """
    Iterates through the data to create supervised learning windows.
    
    Args:
        sequence_length (int): Lookback period (e.g., 24 hours).
        battery_demand (np.array): Demand profile.
        battery_schedule (np.array): Vehicle availability.
        SOC (np.array): State of Charge traces.
        departure_soc, departure_times (np.array): Vehicle metadata.
        static_df (pd.DataFrame): Pre-computed time features.
        env_data (list): List of DataFrames [building, radiation, temp, priceG2B, priceG2V].
        ground_truth (np.array): Optimization targets.
        
    Returns:
        list: List of flattened rows for the dataset.
        int: The number of detected vehicles (for column naming).
    """
    rows = []
    valid_samples = 0
    
    # Unpack environment data for easier access
    b_demand, rad, temp, priceG2B, priceG2V = env_data
    
    # Random index logic from original code (handling negative start indices)
    extended_index = np.random.uniform(
        low=0, high=15287, size=(battery_demand.shape[0])
    ).astype(int)

    # Detect number of vehicles from SOC shape
    n_veh_detected = 0

    for i in range(battery_demand.shape[0]):
        n_window = battery_demand.shape[1] - 1 - sequence_length
        
        # Determine start index
        start_idx = int(battery_demand[i, 0])
        if start_idx < 0:
            start_idx = extended_index[i]
            
        end_idx = start_idx + sequence_length
        hour_now = 23 # As per original logic

        # Prepare Vehicle Data for this specific 'day' (i)
        # Drop NaNs upfront
        sched_row = battery_schedule[i]
        sched_row = sched_row[~np.isnan(sched_row).any(axis=1)]
        
        soc_row = SOC[i].T
        soc_row = soc_row[~np.isnan(soc_row).any(axis=1)]

        V2B_row = V2B[i].T
        V2B_row = V2B_row[~np.isnan(V2B_row).any(axis=1)]
        
        G2V_row = G2V[i].T
        G2V_row = G2V_row[~np.isnan(G2V_row).any(axis=1)]
        
        P2V_row = P2V[i].T
        P2V_row = P2V_row[~np.isnan(P2V_row).any(axis=1)]
        
        optimal_ch = G2V_row + P2V_row - V2B_row
        optimal_ch = optimal_ch.sum(axis=0)  # Sum across vehicles for target comparison

        # --- Sanity Check ---
        if sched_row.shape[0] != soc_row.shape[0]:
            print(f"❌ Mismatch at index {i}: Sched {sched_row.shape} vs SOC {soc_row.shape}")
            continue # Skip bad data instead of crashing immediately if desired

        for n in range(n_window):
            if end_idx > len(b_demand):
                break
            # 1. Slice Environmental Data
            # Note: iloc ranges are [inclusive:exclusive]
            env_feats = [
                b_demand.iloc[start_idx:end_idx, 0].values,
                rad.iloc[start_idx:end_idx, 0].values,
                temp.iloc[start_idx:end_idx, 0].values,
                priceG2B.iloc[start_idx:end_idx, 0].values,
                priceG2V.iloc[start_idx:end_idx, 0].values,
                battery_demand[i, 1 + n: 1 + n + sequence_length] # Battery series
            ]

            # 2. Process Vehicle Data
            avail_flags = sched_row[:, hour_now]
            active_mask = avail_flags > 0
            
            # Slice current timestep data
            curr_soc = soc_row[active_mask, hour_now]
            
            # Get matching departure info
            curr_dep_soc = departure_soc[i][~np.isnan(departure_soc[i])][active_mask]
            curr_dep_time = departure_times[i][~np.isnan(departure_times[i])][active_mask]
            
            # Update detected vehicles for column naming later
            if len(curr_soc) > n_veh_detected:
                n_veh_detected = len(curr_soc)

            # Calculate Priority
            priority = _calculate_priority(curr_soc, curr_dep_soc, curr_dep_time, hour_now)

            # Filter Residual NaNs
            mask_valid = ~(np.isnan(curr_soc) | np.isnan(priority))
            priority = priority[mask_valid]

            # 3. Assemble Row
            # Static (Time) + Environmental (Series) + Vehicle (Priority) + Target
            row_feats = static_df.iloc[end_idx - 1, :].tolist()
            
            for series in env_feats:
                row_feats.extend(series.tolist())
            
            row_feats.extend(np.round(priority, 4).tolist())
            
            # Append Target
            target = float(np.round(optimal_ch[hour_now], 1))
            row_feats.append(target)
            
            rows.append(row_feats)

            # Increment pointers
            start_idx += 1
            end_idx += 1
            hour_now += 1
            valid_samples += 1

    return rows, n_veh_detected

# ==============================================================================
# 3. ORCHESTRATOR
# ==============================================================================

def merge_and_process(
    sequence_length=24, 
    save_feature_info=True, 
    dataset_name="merged_windowed_datasetV3.csv", 
    feature_info_name="feature_infoV3.json", 
    case_id=1
):
    """
    Main entry point. Loads raw data, processes it, and saves the training CSV.
    """
    print(f"\t\t[INFO] Starting Preprocessing for Case {case_id}...")
    
    # --- 1. Load Arrays ---
    base_op_folder = "./data/optimization_results" # Adjust base path as needed
    demand_dir = "./data/battery_demand/"
    case_demand_dir = case_dir(demand_dir, case_id)
    
    battery_demand = np.load(f"{case_demand_dir}/battery_demand.npy")
    battery_details = np.load(f"{case_demand_dir}/battery_details.npy") # [arr_soc, dep_soc, arr_time, dep_time]
    battery_schedule = np.load(f"{case_demand_dir}/battery_availability.npy")
    
    # Optimization results
    opt_path = case_dir(base_op_folder, case_id)
    ground_truth = np.load(f"{opt_path}/optimization/charging_demand.npy")
    
    V2B = np.load(f"{opt_path}/optimization/V2B.npy")
    G2V = np.load(f"{opt_path}/optimization/G2V.npy")
    P2V = np.load(f"{opt_path}/optimization/P2V.npy")
    
    SOC = np.load(f"{opt_path}/optimization/SOC.npy")

    # Unpack details
    departure_soc = battery_details[1]
    departure_times = battery_details[3]

    # --- 2. Load Time Series ---
    ts_dir = "./data/timeseries"
    b_demand = _read_clean_csv(f"{ts_dir}/building_data.csv")
    priceG2B = _read_clean_csv(f"{ts_dir}/electricitycostG2B_data.csv")
    priceG2V = _read_clean_csv(f"{ts_dir}/electricitycostG2V_data.csv")
    rad = _read_clean_csv(f"{ts_dir}/radiation_data.csv")
    temp = _read_clean_csv(f"{ts_dir}/temperature_data.csv")
    
    # Standardize Index
    b_demand.index = pd.to_datetime(b_demand.index)
    static_df = _generate_cyclic_features(b_demand.index)

    # --- 3. Analyze Embeddings (Optional info) ---
    suggested_emb = _analyze_quantization([
        b_demand.values, rad.values, temp.values, priceG2B.values, priceG2V.values, battery_demand[:, 1:]
    ])

    # --- 4. Create Windows ---
    print("\t\t[INFO] Generating sliding windows...")
    rows, n_veh = create_sliding_windows(
        sequence_length=sequence_length,
        battery_demand=battery_demand,
        battery_schedule=battery_schedule,
        SOC=SOC,
        V2B=V2B,
        G2V=G2V,
        P2V=P2V,
        departure_soc=departure_soc,
        departure_times=departure_times,
        static_df=static_df,
        env_data=[b_demand, rad, temp, priceG2B, priceG2V],
        ground_truth=ground_truth
    )

    # --- 5. Format & Save ---
    # Define Column Names
    static_names = ["sin_hour", "cos_hour", "sin_day", "cos_day"]
    series_names = ["building", "radiation", "temperature", "priceG2B", "priceG2V", "battery"]
    ts_cols = [f"{n}_T{t+1}" for n in series_names for t in range(sequence_length)]
    
    # Use detected vehicle count
    batt_cols = [f"priority_V{v+1}" for v in range(n_veh)]
    
    all_cols = static_names + ts_cols + batt_cols + ["target"]
    
    df_processed = pd.DataFrame(rows, columns=all_cols)
    df_processed.to_csv(dataset_name, index=False)
    
    print(f"\t\t[INFO] Saved dataset: {dataset_name} (Shape: {df_processed.shape})")

    if save_feature_info:
        feature_info = {
            "static_cols": static_names,
            "series_blocks": [[f"{n}_T{t+1}" for t in range(sequence_length)] for n in series_names],
            "battery_blocks": [[f"priority_V{v+1}" for v in range(n_veh)]],
            "target_col": "target",
            "num_embeddings": suggested_emb
        }
        with open(feature_info_name, "w") as f:
            json.dump(feature_info, f, indent=4)
        print(f"\t\t[INFO] Saved metadata: {feature_info_name}")

    return df_processed