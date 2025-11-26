import numpy as np
import os
import config as cfg
import numpy as np
import os
import pandas as pd

import numpy as np
import os

def get_battery_details(case_dir):
    """
    Load battery schedule/details from a specific case directory and
    remove NaN padding rows so Gurobi receives clean data.
    
    Parameters:
        case_dir (str): Path to the specific case folder.
    """
    
    # 1. Define Paths
    availability_path = os.path.join(case_dir, "battery_availability.npy")
    details_path      = os.path.join(case_dir, "battery_details.npy")
    demand_path       = os.path.join(case_dir, "battery_demand.npy") 

    if not os.path.exists(availability_path):
        raise FileNotFoundError(f"Could not find data in {case_dir}")

    # 2. Load and Clean Availability (Schedule)
    # Raw shape: (n_samples, max_vehicles, window_size) with NaNs
    available_raw = np.load(availability_path)
    
    # List comprehension to keep only valid rows (vehicles) for each sample
    # We check if a row has ANY NaNs. If yes, it's a padding row.
    available = [
        sample[~np.isnan(sample).any(axis=1)].astype(int) 
        for sample in available_raw
    ]
    
    # 3. Load and Clean Details [SOC_a, SOC_d, t_a, t_d]
    # Details shape: (4, n_samples, max_vehicles)
    details_raw = np.load(details_path, allow_pickle=True)
    
    # Unpack raw arrays
    SOC_a_raw, SOC_d_raw, t_a_raw, t_d_raw = details_raw[0], details_raw[1], details_raw[2], details_raw[3]

    # Clean each one (remove NaNs)
    SOC_a_v = [sample[~np.isnan(sample)] for sample in SOC_a_raw]
    SOC_d_v = [sample[~np.isnan(sample)] for sample in SOC_d_raw]
    t_a_v   = [sample[~np.isnan(sample)].astype(int) for sample in t_a_raw]
    t_d_v   = [sample[~np.isnan(sample)].astype(int) for sample in t_d_raw]

    # 4. Load Index (Timestamps)
    demand_matrix = np.load(demand_path, allow_pickle=True)
    index = demand_matrix[:, 0].astype(int) 

    # 5. Validation
    n_samples = len(index)
    assert len(available) == n_samples, "Mismatch between availability samples and index length."
    
    print(f"\t\t[INFO] Loaded {n_samples} samples from '{case_dir}'")
    
    return index, available, SOC_a_v, SOC_d_v, t_a_v, t_d_v

# def get_battery_details(window_length: int, tolerance: int, Test: bool = False):
#     """
#     Load and merge series (normal + with_nan) and load battery schedule/details.
#     Asserts that lengths match between schedule & details.
    
#     Returns:
#         index: np.ndarray of starting indices (int)
#         available: list of 2D‐arrays (int) of available battery schedules
#         SOC_a_v: list of 1D‐arrays (float) of SOC arrival values
#         SOC_d_v: list of 1D‐arrays (float) of SOC departure values
#         t_a_v: list of 1D‐arrays (int) of arrival times
#         t_d_v: list of 1D‐arrays (int) of departure times
#     """
#     # Construct paths
#     battery_dir = f"./data/battery_demand_tol{cfg.TOLERANCE}/"
    
#     if not Test:
#         schedule_path = os.path.join(battery_dir, f"battery_schedule_{tolerance}nan_window{window_length}.npy")
#         details_path  = os.path.join(battery_dir, f"battery_details_{tolerance}nan_window{window_length}.npy")
#         series_normal_path   = os.path.join(battery_dir, f"battery_series_window{window_length}.npy")
#         series_withnan_path  = os.path.join(battery_dir, f"battery_series_with_{tolerance}nan_window{window_length}.npy")
#     else:
#         schedule_path = os.path.join(battery_dir, f"testdata_battery_schedule_window{window_length}.npy")
#         details_path  = os.path.join(battery_dir, f"testdata_battery_details_window{window_length}.npy")
#         series_normal_path   = os.path.join(battery_dir, f"testdata_battery_series_window{window_length}.npy")
#         series_withnan_path  = None
    
#     # Load schedule (2D/3D array) and filter out NaN rows
#     available_raw = np.load(schedule_path)
#     # Clean each sample: remove trailing rows where any nan
#     available = [sample[~np.isnan(sample).any(axis=1)].astype(int) for sample in available_raw]
    
#     # Load details
#     details = np.load(details_path, allow_pickle=True)
#     SOC_a_v, SOC_d_v, t_a_v, t_d_v = details[0], details[1], details[2], details[3]
#     # Clean: remove nan values
#     SOC_a_v = [sample[~np.isnan(sample)] for sample in SOC_a_v]
#     SOC_d_v = [sample[~np.isnan(sample)] for sample in SOC_d_v]
#     t_a_v   = [sample[~np.isnan(sample)].astype(int) for sample in t_a_v]
#     t_d_v   = [sample[~np.isnan(sample)].astype(int) for sample in t_d_v]
    
#     # Load series starting‐indices
#     index_normal   = np.load(series_normal_path)[:, 0].astype(int)
#     if series_withnan_path is not None:
#         index_withnan  = np.load(series_withnan_path)[:, 0].astype(int)
    
#         # Merge indices
#         index = np.concatenate([index_normal, index_withnan], axis=0)
#     else:
#         index = index_normal
    
#     # Because schedule & details were produced on merged series (I assume),
#     # we expect len(available) == len(SOC_a_v) == len(SOC_d_v) == len(t_a_v) == len(t_d_v) == len(index)
#     n_schedule = len(available)
#     n_details  = len(SOC_a_v)
#     n_index    = len(index)
#     assert n_schedule == n_details == n_index, (
#         f"Length mismatch: schedule={n_schedule}, details={n_details}, index={n_index}"
#     )
    
#     return index, available, SOC_a_v, SOC_d_v, t_a_v, t_d_v