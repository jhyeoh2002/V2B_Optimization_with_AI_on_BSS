from matplotlib import axis
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
# 1. LOGIC ENGINE (ALLOCATION RULES)
# ==========================================
def _calculate_priority(soc_current, dep_soc, dep_time, current_hour):
    """
    Calculates charging priority.
    High Priority = Large gap between current SOC and target SOC with little time left.
    """
    time_remaining = dep_time - current_hour
    # Handle vehicles that have already departed or are just departing (avoid div/0)
    time_remaining = np.maximum(time_remaining, 0.5) 
    
    # Priority Formula
    priority = (dep_soc - soc_current) / time_remaining
    
    # Mask out negative priorities (already charged enough)
    priority = np.maximum(priority, 0.0)
    return priority

def allocate_and_update_state(ai_power_decision, current_socs, avail_mask, dep_socs, dep_times, hour_now):
    """
    Apply constraints, force charge logic, and physics updates.
    """
    # System Constants (Adjust to your real battery specs)
    BATTERY_CAPACITY_KWH = 1.7 
    MAX_CHARGER_POWER = 14.0 # kW per charger
    
    # 1. Identify Active Vehicles
    # We only care about vehicles that are present AND have a target > current
    needs_charge_mask = (current_socs < dep_socs) & (avail_mask > 0)
    
    # 2. Force Charge Logic
    # If time <= 2 hours and we are below target, FORCE MAX CHARGE
    time_left = dep_times - hour_now
    force_charge_mask = needs_charge_mask & (time_left <= 2.0)
    
    # Calculate power needed for force-charge vehicles
    force_power_allocation = np.zeros_like(current_socs)
    force_power_allocation[force_charge_mask] = MAX_CHARGER_POWER
    
    total_force_power = np.sum(force_power_allocation)
    
    # 3. AI Allocation (Remaining Power)
    remaining_power_budget = ai_power_decision - total_force_power
    remaining_power_budget = max(0.0, remaining_power_budget)
    
    # Distribute remaining budget based on Priority
    priorities = _calculate_priority(current_socs, dep_socs, dep_times, hour_now)
    # Only consider those who haven't been force-charged yet
    normal_charge_mask = needs_charge_mask & (~force_charge_mask)
    
    active_priorities = priorities * normal_charge_mask
    sum_prio = np.sum(active_priorities)
    
    normal_power_allocation = np.zeros_like(current_socs)
    
    if sum_prio > 0 and remaining_power_budget > 0:
        # Weighted distribution
        normal_power_allocation = (active_priorities / sum_prio) * remaining_power_budget
    
    # 4. Combine & Apply Physics
    total_power_allocation = force_power_allocation + normal_power_allocation
    
    # Clip individual allocation to charger max limit
    total_power_allocation = np.clip(total_power_allocation, 0, MAX_CHARGER_POWER)
    
    # Update SOC: New = Old + (Power * 1hr) / Capacity
    soc_delta = (total_power_allocation * 1.0) / BATTERY_CAPACITY_KWH
    new_socs = current_socs + soc_delta
    
    # 5. Final Safety Clip (0.0 to 1.0)
    new_socs = np.clip(new_socs, 0.0, 1.0)
    
    executed_power = np.sum(total_power_allocation)
    
    return new_socs, executed_power

# ==========================================
# 2. FEATURE CONSTRUCTION (Dynamic)
# ==========================================
def get_dynamic_input(i, iter, current_socs, b_demand, rad, temp, priceG2B, priceG2V, battery_demand_full, battery_details, battery_schedule, feature_info):
    """
    Constructs input vector using the LIVE `current_socs` passed from the loop.
    Enforces shape matching with the trained model.
    """
    # Time indexing
    start_idx = iter
    end_idx = iter + 24
    
    # 1. Series Features (120 dims)
    env_start = int(battery_demand_full[i, 0]) + iter
    
    batt_demand_window = battery_demand_full[i, 1:][start_idx : end_idx]
    
    series_flat = np.concatenate([
        b_demand.iloc[env_start : env_start + 24, 0].values,
        rad.iloc[env_start : env_start + 24, 0].values,
        temp.iloc[env_start : env_start + 24, 0].values,
        priceG2B.iloc[env_start : env_start + 24, 0].values,
        priceG2V.iloc[env_start : env_start + 24, 0].values,
        batt_demand_window
    ])
    
    # if len(series_flat) != 120:
    #     return None # Boundary check

    # 2. Static Features (4 dims)
    current_time_idx = env_start + 23 
    static_df = _generate_cyclic_features(b_demand.index)
    static_vec = static_df.iloc[current_time_idx].tolist()

    # 3. Vehicle Features (Dynamic part)
    hour_of_day = (23 + iter) % 24
    
    # Get mask for current hour (Safe handling of NaNs)
    sched_row = battery_schedule[i] 
    # Replace NaNs with 0 to preserve shape for reshaping
    sched_row = np.nan_to_num(sched_row, nan=0.0)
    
    # Assuming sched_row is [Vehicles x 48] or [Vehicles x 24] flattened?
    # Ideally sched_row is already 2D [Vehicles, 48]. If it's flattened 1D, reshape:
    if sched_row.ndim == 1:
        sched_row = sched_row.reshape(-1, 48)
        
    avail_flags = sched_row[:, hour_of_day] 
    # print("battery_detail", battery_details.shape)
    # Calculate Priority
    # dep_socs = battery_details[1][i]
    # print(dep_socs.shape)
    # dep_socs = dep_socs[~np.isnan(dep_socs)]  # Clean NaNs
    # dep_times = battery_details[3][i]
    # print(dep_times.shape)

    # dep_times = dep_times[~np.isnan(dep_times)]  # Clean NaNs

    # prio_vec = _calculate_priority(current_socs, dep_socs, dep_times, hour_of_day)
    
    # Zero out priority for unavailable cars
    active_mask = (avail_flags > 0)
    # prio_vec = prio_vec * active_mask
    current_socs = current_socs[active_mask]
    print("current_socs", current_socs.shape)
    
    # # --- DIMENSION FIX: Ensure prio_vec matches model expectation ---
    # expected_veh_dim = sum(len(b) for b in feature_info['battery_blocks'])
    # current_veh_dim = len(prio_vec)
    
    # if current_veh_dim > expected_veh_dim:
    #     # Truncate if we have more vehicles than model expects
    #     prio_vec = prio_vec[:expected_veh_dim]
    # elif current_veh_dim < expected_veh_dim:
    #     # Pad if we have fewer
    #     pad_len = expected_veh_dim - current_veh_dim
    #     prio_vec = np.pad(prio_vec, (0, pad_len), 'constant')
        
    # 4. Concatenate
    full_vector = np.concatenate([static_vec, series_flat, current_socs])
    
    return torch.tensor(full_vector, dtype=torch.float32).unsqueeze(0)

# ==========================================
# 3. STANDARD HELPERS
# ==========================================
def _generate_cyclic_features(index):
    df = pd.DataFrame(index=index)
    df["sin_hour"] = np.sin(2 * np.pi * index.hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * index.hour / 24)
    df["sin_day"] = np.sin(2 * np.pi * index.dayofweek / 7)
    df["cos_day"] = np.cos(2 * np.pi * index.dayofweek / 7)
    return df.round(5)

def _read_clean_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    try: df.index = pd.to_datetime(df.index)
    except: pass
    return df.ffill().bfill()

def load_data_and_model(case_id, run_name):
    base_path = os.path.join(cfg.TRAIN_RESULTS_DIR, run_name)
    with open(os.path.join(base_path, f"feature_info_{run_name}.json"), 'r') as f:
        feat_info = json.load(f)
    
    dims = {
        "static": len(feat_info["static_cols"]),
        "series": len(feat_info["series_blocks"]),
        "seq": len(feat_info["series_blocks"][0]),
        "veh": sum(len(b) for b in feat_info["battery_blocks"]),
        "vocab": 5000
    }
    
    model = STAF(dims["static"], dims["series"], dims["seq"], dims["veh"], dims["vocab"],
                 cfg.EMBEDDING_DIM, cfg.N_HEADS, 256, 64, cfg.DROPOUT).to(cfg.DEVICE)
    model.load_state_dict(torch.load(os.path.join(base_path, "checkpoints", "best_model.pth"), map_location=cfg.DEVICE))
    model.eval()

    # Load Data
    ts_dir = "./data/timeseries"
    b_demand = _read_clean_csv(f"{ts_dir}/building_data.csv")
    priceG2B = _read_clean_csv(f"{ts_dir}/electricitycostG2B_data.csv")
    priceG2V = _read_clean_csv(f"{ts_dir}/electricitycostG2V_data.csv")
    rad = _read_clean_csv(f"{ts_dir}/radiation_data.csv")
    temp = _read_clean_csv(f"{ts_dir}/temperature_data.csv")
    
    # Load Numpy details
    c_dir = case_dir(cfg.BATTERYDEMAND_DIR, 0)
    batt_demand = np.load(f"{c_dir}/battery_demand.npy")
    batt_details = np.load(f"{c_dir}/battery_details.npy")
    batt_sched = np.load(f"{c_dir}/battery_availability.npy")
    
    return model, feat_info, b_demand, rad, temp, priceG2B, priceG2V, batt_demand, batt_details, batt_sched

# ==========================================
# 4. MAIN SIMULATION LOOP
# ==========================================
def run_dynamic_simulation(case_id, run_name):
    print(f"🚀 Starting Dynamic Simulation (Updating SOC per hour)")
    
    # 1. Load Everything
    model, feat_info, b_demand, rad, temp, priceG2B, priceG2V, batt_demand, batt_details, batt_sched = load_data_and_model(case_id, run_name)
    
    # 2. Select Day to Test
    day_idx = 0 
    
    # 3. INITIALIZATION
    print("[INFO] Loading Initial State from Optimization file...")
    SOC_opt = np.load(f"data/optimization_resultsV2/case0_test/optimization/SOC.npy")
    # print("Initial SOC State:", SOC_opt.shape)

    # Get SOCs at the END of the previous period (Hour 23)
    current_soc_state = SOC_opt[day_idx].T[:, 23] 
    
    # FIX: Do NOT remove NaNs, just fill them with 0. 
    # Removing them breaks alignment with 'batt_sched' and the physics loop.
    # print("Initial SOC State:", current_soc_state.shape)
    # current_soc_state = np.nan_to_num(current_soc_state, nan=0.0)
    # current_soc_state = current_soc_state[~np.isnan(current_soc_state)]  # Clean NaNs
    # print("Initial SOC State:", current_soc_state.shape)

    # 4. SIMULATION LOOP (Iter 0 to 23)
    simulation_results = []
    
    for iter in range(24):
        
        # A. BUILD INPUT (Using current_soc_state)
        input_tensor = get_dynamic_input(
            day_idx, iter, current_soc_state, 
            b_demand, rad, temp, priceG2B, priceG2V, batt_demand, batt_details, batt_sched, feat_info
        )
        # print("inputtensor",input_tensor)
        if input_tensor is None: 
            print("⚠️ Input is None, ending simulation early.")
            break

        # B. AI PREDICTION
        with torch.no_grad():
            pred_power_kw = model(input_tensor.to(cfg.DEVICE)).item()
            
        # C. ALLOCATION & UPDATE
        hour_idx = (23 + iter) % 24
        
        # Get availability (Fixing potential nan issues)
        sched_row = np.nan_to_num(batt_sched[day_idx], nan=0.0)
        if sched_row.ndim == 1: sched_row = sched_row.reshape(-1, 48)
        
        avail_mask = (sched_row[:, hour_idx] > 0).astype(float)
        
        target_socs = np.nan_to_num(batt_details[1][day_idx], nan=0.0)
        target_times = np.nan_to_num(batt_details[3][day_idx], nan=0.0)
        
        # *** CRITICAL UPDATE ***
        next_socs, executed_power = allocate_and_update_state(
            pred_power_kw,
            current_soc_state,
            avail_mask,
            target_socs,
            target_times,
            hour_idx
        )
        
        # D. LOGGING
        # Only calc mean for cars currently present
        avg_soc = np.mean(next_socs[avail_mask > 0]) if np.sum(avail_mask) > 0 else 0.0
        print(f"Hour {24+iter}: AI Request={pred_power_kw:.2f}kW | Executed={executed_power:.2f}kW | Avg SOC={avg_soc:.2f}")
        simulation_results.append(executed_power)
        
        # E. STATE TRANSITION
        current_soc_state = next_socs

    print("\n✅ Simulation Complete.")

if __name__ == "__main__":
    run_dynamic_simulation(0, "run_name_here")