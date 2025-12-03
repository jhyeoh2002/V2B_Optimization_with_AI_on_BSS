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
    
    Args:
        ai_power_decision (float): Total power requested by AI (kW).
        current_socs (np.array): Current SOC state of all vehicles [0.0 - 1.0].
        avail_mask (np.array): Boolean mask of vehicles present at site.
        dep_socs (np.array): Target SOCs.
        dep_times (np.array): Departure times.
        hour_now (int): Current simulation hour.
        
    Returns:
        new_socs (np.array): Updated SOCs for next hour.
        executed_power (float): Actual total power used.
    """
    # System Constants (Adjust to your real battery specs)
    BATTERY_CAPACITY_KWH = 60.0 
    MAX_CHARGER_POWER = 7.0 # kW per charger
    
    # 1. Identify Active Vehicles
    # We only care about vehicles that are present AND have a target > current
    needs_charge_mask = (current_socs < dep_socs) & (avail_mask > 0)
    
    # 2. Force Charge Logic
    # If time <= 2 hours and we are below target, FORCE MAX CHARGE
    time_left = dep_times - hour_now
    force_charge_mask = needs_charge_mask & (time_left <= 2.0)
    
    # Calculate power needed for force-charge vehicles
    # Give them MAX_CHARGER_POWER or whatever is needed to hit 100%
    force_power_allocation = np.zeros_like(current_socs)
    force_power_allocation[force_charge_mask] = MAX_CHARGER_POWER
    
    total_force_power = np.sum(force_power_allocation)
    
    # 3. AI Allocation (Remaining Power)
    # The AI decides the generic load, but we subtract what we MUST use for force charging
    remaining_power_budget = ai_power_decision - total_force_power
    remaining_power_budget = max(0.0, remaining_power_budget) # Can't be negative
    
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
def get_dynamic_input(i, iter, current_socs, b_demand, rad, temp, price, battery_demand_full, battery_details, battery_schedule, feature_info):
    """
    Constructs input vector using the LIVE `current_socs` passed from the loop.
    """
    # Time indexing: 
    # iter 0 = Hour 23 (start of sim) -> predicting for next hour
    # We need a 24-hour window ending at the current time.
    
    # 'battery_demand_full' is length 48. 
    # Window start slides: 0->24, 1->25, etc.
    start_idx = iter
    end_idx = iter + 24
    
    # 1. Series Features (120 dims)
    # Slicing the environment data
    env_start = int(battery_demand_full[i, 0]) + iter
    
    # Slicing the Battery Demand history (Sliding Window)
    # We take the row 'i', ignore column 0 (index), and slice the 24h window
    batt_demand_window = battery_demand_full[i, 1:][start_idx : end_idx]
    
    series_flat = np.concatenate([
        b_demand.iloc[env_start : env_start + 24, 0].values,
        rad.iloc[env_start : env_start + 24, 0].values,
        temp.iloc[env_start : env_start + 24, 0].values,
        price.iloc[env_start : env_start + 24, 0].values,
        batt_demand_window
    ])
    
    if len(series_flat) != 120:
        return None # Boundary check

    # 2. Static Features (4 dims)
    # Using the LAST hour of the window as "current time"
    current_time_idx = env_start + 23 
    static_df = _generate_cyclic_features(b_demand.index)
    static_vec = static_df.iloc[current_time_idx].tolist()

    # 3. Vehicle Features (Dynamic part)
    # Current hour in 0-23 format for availability checks
    hour_of_day = (23 + iter) % 24
    
    # Get mask for current hour
    sched_row = battery_schedule[i] # [Vehicles, 24] or [Vehicles, 48]? Assuming fit to day
    sched_row = sched_row[~np.isnan(sched_row)].reshape(-1, 48)
    # Assuming sched_row aligns with the specific day we are simulating
    avail_flags = sched_row[:, hour_of_day] 
    
    # Calculate Priority dynamically based on the passed-in SOCs
    dep_socs = battery_details[1][i]
    dep_socs = dep_socs[~np.isnan(dep_socs)]
    dep_times = battery_details[3][i]
    dep_times = dep_times[~np.isnan(dep_times)]

    # IMPORTANT: Ensure NaN handling if vehicles aren't in this case
    # This generates the vector of size [N_VEHICLES]
    prio_vec = _calculate_priority(current_socs, dep_socs, dep_times, hour_of_day)
    
    # Zero out priority for unavailable cars so model ignores them
    active_mask = (avail_flags > 0)
    prio_vec = prio_vec * active_mask
    
    # 4. Concatenate
    full_vector = np.concatenate([static_vec, series_flat, prio_vec])
    
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
    # Load Model (Same as your logic)
    base_path = os.path.join(cfg.TRAIN_RESULTS_DIR, run_name)
    with open(os.path.join(base_path, f"feature_info_{run_name}.json"), 'r') as f:
        feat_info = json.load(f)
    
    dims = {
        "static": len(feat_info["static_cols"]),
        "series": len(feat_info["series_blocks"]),
        "seq": len(feat_info["series_blocks"][0]),
        "veh": sum(len(b) for b in feat_info["battery_blocks"]),
        "vocab": feat_info.get("num_embeddings", cfg.NUM_EMBEDDINGS)
    }
    
    model = STAF(dims["static"], dims["series"], dims["seq"], dims["veh"], dims["vocab"],
                 cfg.EMBEDDING_DIM, cfg.N_HEADS, cfg.HIDDEN_DIM_1, cfg.HIDDEN_DIM_2, cfg.DROPOUT).to(cfg.DEVICE)
    model.load_state_dict(torch.load(os.path.join(base_path, "checkpoints", "best_model.pth"), map_location=cfg.DEVICE))
    model.eval()

    # Load Data
    ts_dir = "./data/timeseries"
    b_demand = _read_clean_csv(f"{ts_dir}/building_data.csv")
    price = _read_clean_csv(f"{ts_dir}/electricitycostG2B_data.csv")
    rad = _read_clean_csv(f"{ts_dir}/radiation_data.csv")
    temp = _read_clean_csv(f"{ts_dir}/temperature_data.csv")
    
    # Load Numpy details
    c_dir = case_dir(cfg.BATTERYDEMAND_DIR, 0)
    batt_demand = np.load(f"{c_dir}/battery_demand.npy")
    batt_details = np.load(f"{c_dir}/battery_details.npy")
    batt_sched = np.load(f"{c_dir}/battery_availability.npy")
    
    return model, feat_info, b_demand, rad, temp, price, batt_demand, batt_details, batt_sched

# ==========================================
# 4. MAIN SIMULATION LOOP
# ==========================================
def run_dynamic_simulation(case_id, run_name):
    print(f"🚀 Starting Dynamic Simulation (Updating SOC per hour)")
    
    # 1. Load Everything
    model, feat_info, b_demand, rad, temp, price, batt_demand, batt_details, batt_sched = load_data_and_model(case_id, run_name)
    
    # 2. Select Day to Test
    day_idx = 0 
    
    # 3. INITIALIZATION (The "Reference" Step)
    # We read the SOC state at Hour 23 (index 23) from the optimized file
    # This is our starting point for the simulation.
    print("[INFO] Loading Initial State from Optimization file...")
    SOC_opt = np.load(f"data/optimization_results/case0_test/optimization/SOC.npy")
    
    # SOC_opt shape: [Days, Vehicles, Hours]? Or [Days, Hours, Vehicles]? 
    # Assuming shape [Days, Vehicles, 24] based on your code `soc_row = SOC[i].T`
    
    # Get SOCs at the END of the previous period (Hour 23)
    # This becomes the starting SOC for our simulation loop (Iter 0)
    current_soc_state = SOC_opt[day_idx].T[:, 23] 
    # print(len(current_soc_state))

    # print(current_soc_state)
    
    # Handle NaNs (vehicles not there)
    current_soc_state = current_soc_state[~np.isnan(current_soc_state)]
    # print(len(current_soc_state))

    # print(current_soc_state)
    
    # 4. SIMULATION LOOP (Iter 0 to 23)
    # Iter 0 corresponds to prediction for Hour 0 of the NEXT day (or Hour 24 contiguous)
    simulation_results = []
    
    for iter in range(24):
        # Current Global Hour (23 + iter -> 23, 24, 25...)
        # Wait, if iter 0 is the first prediction, we are at hour 23 looking forward.
        
        # A. BUILD INPUT (Using current_soc_state)
        input_tensor = get_dynamic_input(
            day_idx, iter, current_soc_state, 
            b_demand, rad, temp, price, batt_demand, batt_details, batt_sched, feat_info
        )
        
        if input_tensor is None: break

        # B. AI PREDICTION
        with torch.no_grad():
            pred_power_kw = model(input_tensor.to(cfg.DEVICE)).item()
            
        # C. ALLOCATION & UPDATE (The Feedback Loop)
        # Get details for this hour
        hour_idx = (23 + iter) % 24 # Wrap around 0-23 if needed, or keep continuous
        
        # Get availability for the moment
        sched_row = batt_sched[day_idx]
        avail_mask = (sched_row[:, hour_idx] > 0).astype(float)
        
        target_socs = batt_details[1][day_idx]
        target_socs = target_socs[~np.isnan(target_socs)]
        target_times = batt_details[3][day_idx]
        target_times = target_times[~np.isnan(target_times)]
        
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
        print(f"Hour {24+iter}: AI Request={pred_power_kw:.2f}kW | Executed={executed_power:.2f}kW | Avg SOC={np.mean(next_socs[avail_mask>0]):.2f}")
        simulation_results.append(executed_power)
        
        # E. STATE TRANSITION
        # The output of this hour becomes the input for the next iter
        current_soc_state = next_socs

    print("\n✅ Simulation Complete.")

if __name__ == "__main__":
    run_dynamic_simulation(0, "run_name_here")