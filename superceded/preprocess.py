import os
import sys
import numpy as np
import pandas as pd
import json
from itertools import chain
from util.case_dir import case_dir
# ==== Path setup ====
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
# DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# OUTPUT_OPT_DIR = os.path.join(PROJECT_ROOT, "optimization", "output")

sys.path.append(os.path.abspath(".."))


def merge_and_process(sequence_length=24, save_feature_info=True, tolerance=4, window_size=48, optimization_folder="WL48_PV500_48H_with_2nan", dataset_name="merged_windowed_datasetV3.csv", feature_info_name="feature_infoV3.json", case_id=1):
    """Merge npy + CSV time-series into one supervised learning dataset."""
    
    demand_dir = f"./data/battery_demand/"
    case_demand_dir = case_dir(demand_dir, case_id)
    
    # === Load npy arrays ===
    battery_demand = np.load( f"{case_demand_dir}/battery_demand.npy")
    battery_details = np.load(f"{case_demand_dir}/battery_details.npy")
    battery_schedule = np.load(f"{case_demand_dir}/battery_availability.npy")  # availability schedule

    arrival_soc = battery_details[0]
    departure_soc = battery_details[1]
    arrival_times = battery_details[2]
    departure_times = battery_details[3]

    ground_truth = np.load(f"{optimization_folder}/optimization/charging_demand.npy")
    SOC = np.load(f"{optimization_folder}/optimization/SOC.npy")

    # === Load CSV data ===
    def read_clean_csv(path):
        df = pd.read_csv(path, index_col=0)
        df = df.apply(lambda c: c.astype(str).str.replace('"', '').str.strip())
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    building_demand = read_clean_csv("./data/timeseries/building_data.csv")
    electricity_price = read_clean_csv("./data/timeseries/electricitycostG2B_data.csv")
    radiation = read_clean_csv("./data/timeseries/radiation_data.csv")
    temperature = read_clean_csv("./data/timeseries/temperature_data.csv")

    # === Static features ===
    building_demand.index = pd.to_datetime(building_demand.index)
    static_df = pd.DataFrame(index=building_demand.index)
    static_df["sin_hour"] = np.sin(2 * np.pi * building_demand.index.hour / 24)
    static_df["cos_hour"] = np.cos(2 * np.pi * building_demand.index.hour / 24)
    static_df["sin_day"] = np.sin(2 * np.pi * building_demand.index.dayofweek / 7)
    static_df["cos_day"] = np.cos(2 * np.pi * building_demand.index.dayofweek / 7)
    static_df = static_df.round(5)

    # === Quantization statistics (for embedding) ===
    all_series = list(chain(
        building_demand.values.flatten(),
        radiation.values.flatten(),
        temperature.values.flatten(),
        electricity_price.values.flatten(),
        battery_demand[:, 1:].flatten()
    ))
    indices = np.round(np.array(all_series) * 1000) + 5000
    indices = np.clip(indices, 0, None).astype(int)
    num_unique = len(np.unique(indices))
    suggested_num_embeddings = int(num_unique * 1.2)
    print(f"\t\t[INFO] Unique quantized indices: {num_unique}")
    print(f"\t\t[INFO] Suggested num_embeddings: {suggested_num_embeddings}")

    # === Determine n_veh from existing data ===
    n_veh = SOC.shape[2] if SOC.ndim == 3 else SOC.shape[1]

    assert battery_schedule.shape[0] == battery_demand.shape[0] == ground_truth.shape[0] == SOC.shape[0], f"Mismatched sample counts among datasets. {battery_schedule.shape[0]} != {battery_demand.shape[0]} != {ground_truth.shape[0]} != {SOC.shape[0]}"

    # === Build dataset ===
    rows = []
    valid_series = 0
    valid_samples = 0
    extended_index = np.random.uniform(low=0, high=15287, size=(battery_demand.shape[0])).astype(int)
    
    for i in range(battery_demand.shape[0]):
        
        n_window = battery_demand.shape[1] - 1 - sequence_length
        
        start_idx = int(battery_demand[i, 0]) if battery_demand[i, 0] >= 0 else extended_index[i]
        end_idx = start_idx + sequence_length
        
        hour_now = 23
        
        # === Vehicle-level data ===
        sched = battery_schedule[i].copy()          # (n_veh, window_len)
        sched = sched[~np.isnan(sched).any(axis=1)] # drop NaN rows
        SOC_row = SOC[i].copy().T
        SOC_row = SOC_row[~np.isnan(SOC_row).any(axis=1)]  # same filtering
        
        for window in range(n_window):
            
            

            if end_idx > len(building_demand):
                continue

            # === Temporal 24h slices ===
            building_window = building_demand.iloc[start_idx:end_idx, 0].values
            radiation_window = radiation.iloc[start_idx:end_idx, 0].values
            temperature_window = temperature.iloc[start_idx:end_idx, 0].values
            price_window = electricity_price.iloc[start_idx:end_idx, 0].values
            
            battery_window = battery_demand[i, 1 + window:1 + window + sequence_length]




            # If still mismatched, log difference and break early for inspection
            if sched.shape[0] != SOC_row.shape[0]:
                print(f"❌ Mismatch detected at index {i}")
                diff = abs(sched.shape[0] - SOC_row.shape[0])
                print(f"  Difference in vehicle count: {diff}")
                print("  Possible cause: truncated NaN filtering or schedule padding.")
                np.save(f"debug_sched_{i}.npy", sched)
                np.save(f"debug_soc_{i}.npy", SOC_row)
                raise ValueError(f"[Index {i}] schedule/SOC mismatch: {sched.shape} vs {SOC_row.shape}")

            if sched.shape[0] != SOC_row.shape[0]:
                raise ValueError(f"[Index {i}] schedule/SOC mismatch: {sched.shape} vs {SOC_row.shape}")

            avail_flags = sched[:, hour_now]  # last hour availability flag
            active_mask = avail_flags > 0

            SOC_current = SOC_row[active_mask, hour_now]  # final SOC at window end
            departure_soc_dropna = departure_soc[i][~np.isnan(departure_soc[i])]
            departure_times_dropna = departure_times[i][~np.isnan(departure_times[i])]

            dep_soc_current = departure_soc_dropna[active_mask]
            dep_time_current = departure_times_dropna[active_mask]

            if not (len(SOC_current) == len(dep_soc_current) == len(dep_time_current)):
                raise ValueError(f"[Index {i}] inconsistent vehicle lengths")

            # === Compute priority safely ===
            denom = (dep_time_current - hour_now) # hours until departure from window end
            denom[denom == 0] = np.nan
            priority = (dep_soc_current - SOC_current) / (denom + 1e-6)

            # Drop any residual NaNs
            mask_valid = ~(np.isnan(SOC_current) | np.isnan(dep_soc_current) | np.isnan(dep_time_current) | np.isnan(priority))
            SOC_current = SOC_current[mask_valid]
            dep_soc_current = dep_soc_current[mask_valid]
            dep_time_current = dep_time_current[mask_valid]
            priority = priority[mask_valid]

            # === Assemble features ===
            row_feats = static_df.iloc[end_idx - 1, :].tolist()

            for series in [building_window, radiation_window, temperature_window, price_window, battery_window]:
                row_feats.extend(series.tolist())

            for block in [priority]:
                row_feats.extend(np.round(block,4).tolist())

            target = float(np.round(ground_truth[i, sequence_length], 3))
            row_feats.append(target)
            rows.append(row_feats)
            
            ## update indices
            start_idx += 1
            end_idx += 1
            hour_now += 1
            
            valid_samples += 1
            
        valid_series += 1
    
    print(f"\t\t[INFO] Processed {valid_series} series with {valid_samples} valid samples.")

    # === Column naming ===
    static_names = ["sin_hour", "cos_hour", "sin_day", "cos_day"]
    ts_names = [f"{name}_T{t+1}" for name in ["building", "radiation", "temperature", "price", "battery"] for t in range(sequence_length)]
    # Use vehicle count from first valid sample
    n_veh_detected = len(SOC_current)
    # batt_names = [f"{name}_V{v+1}" for name in ["SOC_current", "departure_soc", "departure_time", "priority"] for v in range(n_veh_detected)]
    batt_names = [f"{name}_V{v+1}" for name in ["priority"] for v in range(n_veh_detected)]

    cols = static_names + ts_names + batt_names + ["target"]

    # === Build DataFrame ===
    df_processed = pd.DataFrame(rows, columns=cols)

    # === Save outputs ===
    # output_path = os.path.join(BASE_DIRdataset_name)
    df_processed.to_csv(dataset_name, index=False)
    print(f"\t\t[INFO] Saved merged dataset: {dataset_name}  (shape: {df_processed.shape})")

    if save_feature_info:
        feature_info = {
            "static_cols": static_names,
            "series_blocks": [[f"{n}_T{t+1}" for t in range(sequence_length)] for n in ["building", "radiation", "temperature", "price", "battery"]],
            "battery_blocks": [[f"{n}_V{v+1}" for v in range(n_veh_detected)] for n in ["priority"]],
            "target_col": "target",
            "num_embeddings": suggested_num_embeddings
        }
        # json_path = os.path.join(BASE_DIR, feature_info_name)
        with open(feature_info_name, "w") as f:
            json.dump(feature_info, f, indent=4)
        print(f"\t\t[INFO] Saved feature metadata: {feature_info_name}")

    return df_processed


