from heapq import merge
import os
import sys
import numpy as np
import pandas as pd
import json

# ==== Path setup ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_OPT_DIR = os.path.join(PROJECT_ROOT, "optimization", "output")

sys.path.append(os.path.abspath(".."))

def merge_and_process(sequence_length=24, save_feature_info=True):
    """Merge npy + CSV time-series into one supervised learning dataset."""

    # === Load npy arrays ===
    battery_demand = np.load(os.path.join(DATA_PROCESSED_DIR, "battery_series_window48.npy"))  # (N, 25)
    ground_truth = np.load(os.path.join(OUTPUT_OPT_DIR, "V1_WL48_PV400/charging_demands.npy"))

    # === Load CSV data (cleaning numeric) ===
    def read_clean_csv(path):
        df = pd.read_csv(path, index_col=0)
        df = df.apply(lambda c: c.astype(str).str.replace('"', '').str.strip())
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    building_demand = read_clean_csv(os.path.join(DATA_PROCESSED_DIR, "building_data.csv"))
    electricity_price = read_clean_csv(os.path.join(DATA_PROCESSED_DIR, "electricitycostG2B_data.csv"))
    radiation = read_clean_csv(os.path.join(DATA_PROCESSED_DIR, "radiation_data.csv"))
    temperature = read_clean_csv(os.path.join(DATA_PROCESSED_DIR, "temperature_data.csv"))

    # === Static features (from datetime index) ===
    building_demand.index = pd.to_datetime(building_demand.index)
    static_df = pd.DataFrame(index=building_demand.index)
    static_df["sin_hour"] = np.sin(2 * np.pi * building_demand.index.hour / 24)
    static_df["cos_hour"] = np.cos(2 * np.pi * building_demand.index.hour / 24)
    static_df["sin_day"]  = np.sin(2 * np.pi * building_demand.index.dayofweek / 7)
    static_df["cos_day"]  = np.cos(2 * np.pi * building_demand.index.dayofweek / 7)
    
    print(static_df.head())
    static_df = static_df.round(5)
    print(static_df.head())

    # === Build merged rows ===
    rows = []
    valid_count = 0
    for i in range(battery_demand.shape[0]):
        start_idx = int(battery_demand[i, 0])
        end_idx = start_idx + sequence_length
        if end_idx > len(building_demand):
            continue  # skip incomplete windows at end of file

        # Extract 24-hour slices
        building_window = building_demand.iloc[start_idx:end_idx, 0].values
        radiation_window = radiation.iloc[start_idx:end_idx, 0].values
        temperature_window = temperature.iloc[start_idx:end_idx, 0].values
        price_window = electricity_price.iloc[start_idx:end_idx, 0].values
        battery_window = battery_demand[i, 1:1 + sequence_length]  # from npy

        # Static features at window end (or start)
        row_feats = static_df.iloc[end_idx - 1, :].tolist()

        # Concatenate all temporal features
        for w in [building_window, radiation_window, temperature_window, price_window, battery_window]:
            row_feats.extend(w.tolist())

        # Target (ground truth)
        target = ground_truth[i, 0].round(3)
        row_feats.append(float(target))
        rows.append(row_feats)
        valid_count += 1

    print(f"✅ Processed {valid_count} valid samples.")

    # === Column naming ===
    static_names = ["sin_hour", "cos_hour", "sin_day", "cos_day"]
    ts_names = ["building", "radiation", "temperature", "price", "battery"]
    ts_cols = [f"{name}_{t}" for name in ts_names for t in range(sequence_length)]
    cols = static_names + ts_cols + ["target"]

    df_processed = pd.DataFrame(rows, columns=cols)

    # === Save merged dataset ===
    output_path = os.path.join(BASE_DIR, "merged_windowed_dataset.csv")
    df_processed.to_csv(output_path, index=False)
    print(f"✅ Saved merged dataset: {output_path}")
    print(f"   → shape: {df_processed.shape}")

    # === Save feature info ===
    if save_feature_info:
        feature_info = {
            "static_cols": static_names,
            "series_blocks": [[f"{n}_{t}" for t in range(sequence_length)] for n in ts_names],
            "target_col": "target"
        }
        json_path = os.path.join(BASE_DIR, "feature_info.json")
        with open(json_path, "w") as f:
            json.dump(feature_info, f, indent=4)
        print(f"✅ Saved feature metadata: {json_path}")

    return df_processed


merge_and_process()