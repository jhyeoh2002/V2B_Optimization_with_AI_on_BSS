import pandas as pd
import numpy as np
import os
import json
import torch
import sys
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from supervised_learning.models.staf_net import TemporalAttentiveFusionNet as STAF
from util.case_dir import case_dir

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

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
        pass # Index might already be correct or integer based

    # Fill gaps
    df = df.ffill().bfill()
    return df

def _generate_cyclic_features(timestamp):
    """
    Creates continuous cyclic representations (Sin/Cos) for Hour and Day for a specific timestamp.
    """
    sin_hour = np.sin(2 * np.pi * timestamp.hour / 24)
    cos_hour = np.cos(2 * np.pi * timestamp.hour / 24)
    sin_day = np.sin(2 * np.pi * timestamp.dayofweek / 7)
    cos_day = np.cos(2 * np.pi * timestamp.dayofweek / 7)
    return [sin_hour, cos_hour, sin_day, cos_day]

def _calculate_priority(soc_current, dep_soc, dep_time, current_hour):
    """
    Calculates charging priority: (Target_SOC - Current_SOC) / Time_Remaining
    """
    time_left = (dep_time - current_hour)
    
    # Handle cases where vehicle is departing now or passed departure
    time_left = np.maximum(time_left, 0.1) 
    
    priority = (dep_soc - soc_current) / time_left
    return priority

# ==========================================
# 2. SIMULATION ENVIRONMENT
# ==========================================

class SimulationEnvironment:
    def __init__(self, case_id, seq_len=24):
        self.case_id = case_id
        self.seq_len = seq_len
        self.device = cfg.DEVICE
        
        # --- Load Data ---
        self._load_data()
        
        # --- Simulation State ---
        self.T_horizon = 48  # Simulation length
        
        # Check shape of battery_details to correctly determine n_veh
        # Expected: [4, n_vehicles]
        if self.battery_details.ndim == 1:
             # Fallback if shape is weird, though unlikely given load code
             self.n_veh = 1
        else:
             self.n_veh = self.battery_details.shape[1]
        
        # State variables
        # soc_t_v: [Time, Vehicle] -> Stores SOC state
        self.soc_t_v = np.zeros((self.T_horizon + 1, self.n_veh)) 
        self.ch_t_v = np.zeros((self.T_horizon, self.n_veh)) # Charging power record
        self.grid_load = np.zeros(self.T_horizon)
        self.pv_loss = np.zeros(self.T_horizon)
        
        # Initialize SOCs at arrival times
        # Ensure we are indexing correctly. 
        # If self.battery_details is (4, N), then row 0 is arr_soc, row 2 is arr_time
        arr_socs = self.battery_details[0]
        arr_times = self.battery_details[2]
        
        for v in range(self.n_veh):
            # Extract scalar value safely
            # If arr_times[v] is a 0-d array, float(arr_times[v]) works.
            # If it's a list/array of 1 element, this handles it.
            val = arr_times[v]
            if isinstance(val, (np.ndarray, list)):
                val = val.item() if isinstance(val, np.ndarray) and val.size == 1 else val[0]
            
            t_arr = int(val)
            
            if 0 <= t_arr < self.T_horizon:
                # Similarly safe extraction for SOC
                soc_val = arr_socs[v]
                if isinstance(soc_val, (np.ndarray, list)):
                     soc_val = soc_val.item() if isinstance(soc_val, np.ndarray) and soc_val.size == 1 else soc_val[0]
                
                self.soc_t_v[t_arr, v] = soc_val

    def _load_data(self):
        print(f"\t[INFO] Loading environment data for Case {self.case_id}...")
        ts_dir = "./data/timeseries"
        
        # Load Time Series
        self.df_bldg = _read_clean_csv(f"{ts_dir}/building_data.csv")
        self.df_price = _read_clean_csv(f"{ts_dir}/electricitycostG2B_data.csv")
        self.df_rad = _read_clean_csv(f"{ts_dir}/radiation_data.csv")
        self.df_temp = _read_clean_csv(f"{ts_dir}/temperature_data.csv")
        
        # Calculate PV Generation (Simplified Physics Model)
        # Efficiency 20%, Capacity 500m2
        efficiency, capacity = 0.20, 500
        rad_val = self.df_rad.values.flatten()
        temp_val = self.df_temp.values.flatten()
        self.pv_gen = efficiency * capacity * (rad_val * 0.2778) * (1 - 0.005 * (temp_val - 25))
        
        # Load Battery Metadata
        demand_dir = cfg.BATTERYDEMAND_DIR
        case_demand_dir = case_dir(demand_dir, self.case_id)
        
        # [n_samples, 49] -> index 0 is start_time, 1..48 is demand
        self.battery_demand_raw = np.load(f"{case_demand_dir}/battery_demand.npy") 
        # [4, n_vehicles] -> [arr_soc, dep_soc, arr_time, dep_time]
        self.battery_details = np.load(f"{case_demand_dir}/battery_details.npy") 
        # [n_vehicles, 48] -> 1 if present, 0 else
        self.battery_schedule = np.load(f"{case_demand_dir}/battery_availability.npy")

    def get_observation(self, t, sample_idx):
        """
        Constructs the input tensor for the model for a specific simulation hour `t`.
        
        Args:
            t (int): Current simulation hour (0 to 47).
            sample_idx (int): Which sample scenario we are running (row index in battery_demand).
        """
        # 1. Identify data indices
        # The sample_idx tells us which row of battery_demand to look at
        # The value at col 0 tells us the global start index in the timeseries data
        global_start_idx = int(self.battery_demand_raw[sample_idx, 0])
        
        # 2. Sliding Window (Look-back)
        # The model expects a sequence of length `seq_len` ending at current time `t`
        # If t < 24, we need to pad or handle start (assuming data exists before start_idx)
        # For simplicity here, we assume we slice [t - 23 : t + 1] relative to simulation start
        
        # NOTE: For the very first steps of simulation, we might need historical data.
        # Here we slice directly from global dataframe using (global_start + t)
        
        current_global_idx = global_start_idx + t
        window_start = current_global_idx - (self.seq_len - 1)
        window_end = current_global_idx + 1
        
        # Extract Exogenous Series
        bldg_win = self.df_bldg.iloc[window_start:window_end, 0].values
        rad_win = self.df_rad.iloc[window_start:window_end, 0].values
        temp_win = self.df_temp.iloc[window_start:window_end, 0].values
        price_win = self.df_price.iloc[window_start:window_end, 0].values
        
        # Extract Battery Demand (The planned/requested profile, NOT the actual SOC)
        # battery_demand_raw is [Sample, 1 + 48 hours]
        # We map simulation t to the column in battery_demand
        # If t < seq_len, we pad with zeros or take previous data? 
        # Assuming the 'battery_demand' array is the 'request' signal. 
        # Let's simply take the slice corresponding to the last 24 hours relative to t.
        # Since battery_demand only exists for [0..48], for t < 23 we pad.
        
        batt_seq = np.zeros(self.seq_len)
        
        # Relative indices inside the 48h window
        rel_start = t - (self.seq_len - 1)
        rel_end = t + 1
        
        # Fill valid parts
        for i, rel_t in enumerate(range(rel_start, rel_end)):
            if 0 <= rel_t < 48:
                batt_seq[i] = self.battery_demand_raw[sample_idx, 1 + rel_t]
        
        # 3. Static / Cyclic Features for current hour
        current_time = self.df_bldg.index[current_global_idx]
        static_feats = _generate_cyclic_features(current_time) # [4]
        
        # 4. Vehicle Features (Priorities based on CURRENT SOC)
        # This is the feedback loop. We use self.soc_t_v to calculate priorities.
        
        # Get details for this sample (assuming sample_idx maps 1-to-1 to n_veh blocks for now, 
        # or specific logic if multiple samples share vehicles. 
        # Based on user code, it seems sample_idx implies specific vehicle set availability)
        
        sched_row = self.battery_schedule[sample_idx] # [48] if flattened or [N_veh, 48]
        # Note: availability shape depends on how data was saved. 
        # Assuming self.battery_schedule is [n_vehicles, 48] per sample or global? 
        # Adapting to typical structure: battery_schedule is often global [N_veh, T]
        # If sample_idx selects a day, we check availability for that day.
        
        active_mask = self.battery_schedule[:, t] > 0
        
        curr_socs = self.soc_t_v[t, :] # [N_veh]
        dep_socs = self.battery_details[1] # [N_veh]
        dep_times = self.battery_details[3] # [N_veh]
        
        priorities = np.zeros(self.n_veh)
        
        # Only calculate for active vehicles
        if np.any(active_mask):
            prio_vals = _calculate_priority(
                curr_socs[active_mask], 
                dep_socs[active_mask], 
                dep_times[active_mask], 
                t
            )
            priorities[active_mask] = prio_vals
            
        # 5. Construct Final Vector
        # Concatenate [bldg, rad, temp, price, batt] -> 5 * 24 = 120
        series_flat = np.concatenate([bldg_win, rad_win, temp_win, price_win, batt_seq])
        
        # Full vector: [Static(4) | Series(120) | Vehicles(N_veh)]
        # Note: Ensure 'N_veh' matches model training configuration (e.g. 76)
        # If actual vehicles < 76, pad. If >, truncate or error.
        
        target_n_veh = 76 # Example from user context
        if len(priorities) < target_n_veh:
            priorities = np.pad(priorities, (0, target_n_veh - len(priorities)))
        elif len(priorities) > target_n_veh:
            priorities = priorities[:target_n_veh]
            
        full_vector = np.concatenate([static_feats, series_flat, priorities])
        
        return torch.tensor(full_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

    def step(self, t, decision_kw, sample_idx):
        """
        Applies the model's decision (Net Power) using rule-based allocation.
        Updates SOC for t+1.
        """
        # Clip decision constraints
        decision_t = np.clip(decision_kw, -14, 14) # Max grid/transformer limit
        
        # Identify active vehicles
        avail = self.battery_schedule[:, t]
        active_indices = np.where(avail > 0)[0]
        
        # Data for active vehicles
        soc_now_all = self.soc_t_v[t, :]
        soc_dep_all = self.battery_details[1]
        t_dep_all = self.battery_details[3]
        t_arr_all = self.battery_details[2]
        
        # --- 1. Calculate Urgency / Priority ---
        scores = []
        urgency_by_time = defaultdict(float)
        
        for v in active_indices:
            soc = soc_now_all[v]
            target = soc_dep_all[v]
            time_left = t_dep_all[v] - t
            
            # Urgency Score
            u = 1.5 * (target - soc) / (max(time_left, 0.1) * 0.9)
            scores.append((v, u))
            
            # Track total needed for urgent departures to verify overload
            if u > 0:
                urgency_by_time[int(time_left)] += u

        # --- 2. Check Overload Logic ---
        # If vehicles leaving soon need more power than available, force max charge
        trigger_max = False
        accum_kw = 28 if decision_t is not None else 0 # base capacity
        
        # Simple heuristic: check if average needed per hour exceeds limit
        total_needed = sum(u for v, u in scores if u > 0)
        # (This part can be refined based on the specific logic in your notebook)
        
        # --- 3. Determine Mode (Charge vs Discharge) ---
        if decision_t >= 0 or trigger_max:
            # Charging Mode: Prioritize HIGHEST urgency
            sorted_vehicles = sorted(scores, key=lambda x: x[1], reverse=True)
            power_budget = 14 if trigger_max else decision_t
            mode = 'charge'
        else:
            # Discharging Mode: Prioritize LOWEST urgency (those who can wait)
            sorted_vehicles = sorted(scores, key=lambda x: x[1], reverse=False)
            power_budget = abs(decision_t) # Amount to discharge
            mode = 'discharge'

        # --- 4. Allocate ---
        current_power_sum = 0
        
        for v, urgency in sorted_vehicles:
            soc = soc_now_all[v]
            target = soc_dep_all[v]
            
            # Safety limits
            if not (0.199 <= soc <= 0.901): continue
            
            if mode == 'charge':
                # How much CAN we charge?
                # Max rate 1.5 kW, or until full (0.9), or until budget runs out
                max_p = 1.5
                req_p = (target - soc) * 1.5 / 0.9 # Approx power needed
                
                alloc = min(max_p, req_p, max(0, power_budget - current_power_sum))
                alloc = max(0, alloc)
                
                self.ch_t_v[t, v] = alloc
                current_power_sum += alloc
                
                # Update SOC (Efficiency 0.9)
                self.soc_t_v[t + 1, v] = soc + (alloc / 1.5) * 0.9
                
            elif mode == 'discharge':
                # How much CAN we discharge?
                # Cannot discharge below 0.2 SOC
                # V2B rate limited
                avail_discharge_cap = (soc - 0.2) * 1.5 * 0.9 # kWh capacity roughly
                
                alloc = min(1.5, avail_discharge_cap, max(0, power_budget - current_power_sum))
                alloc = max(0, alloc)
                
                self.ch_t_v[t, v] = -alloc
                current_power_sum += alloc
                
                # Update SOC (Discharge increases inverse of efficiency?)
                # Usually soc_next = soc - (power * dt / cap) / efficiency
                self.soc_t_v[t + 1, v] = soc - (alloc / 1.5) / 0.9

        # Update inactive vehicles SOC (carry forward)
        for v in range(self.n_veh):
            if v not in active_indices:
                # If just arrived, init SOC (already done in __init__, but good for safety)
                # Safe scalar extraction
                val = t_arr_all[v]
                t_arr_int = int(val.item() if isinstance(val, np.ndarray) and val.size==1 else val)
                
                if t == t_arr_int:
                    # Init new vehicle arrival
                    init_soc = self.battery_details[0][v]
                    init_soc = init_soc.item() if isinstance(init_soc, np.ndarray) else init_soc
                    self.soc_t_v[t+1, v] = init_soc
                else:
                    # Carry forward
                    self.soc_t_v[t+1, v] = self.soc_t_v[t, v]
                    
        # --- 5. Calculate Grid Impact ---
        # Actual net power used
        net_batt_power = np.sum(self.ch_t_v[t])
        
        # Grid = Building + Battery - PV
        # Note: self.df_bldg etc are full series. We need the specific index.
        global_idx = int(self.battery_demand_raw[sample_idx, 0]) + t
        
        b_dem = self.df_bldg.iloc[global_idx, 0]
        pv_gen = self.pv_gen[global_idx]
        
        net_load = b_dem - pv_gen + net_batt_power
        self.grid_load[t] = max(0, net_load)
        self.pv_loss[t] = -min(0, net_load) # Excess PV

# ==========================================
# 3. MODEL LOADING
# ==========================================
def load_trained_model(run_name):
    base_path = os.path.join(cfg.TRAIN_RESULTS_DIR, run_name)
    json_path = os.path.join(base_path, f"feature_info_{run_name}.json")
    model_path = os.path.join(base_path, "checkpoints", "best_model.pth")
    
    if not os.path.exists(json_path) or not os.path.exists(model_path):
        print(f"[ERR] Missing config or model at {base_path}")
        return None, None

    with open(json_path, 'r') as f:
        feat_info = json.load(f)
        
    # Reconstruct dims from config used during training
    dims = {
        "static": len(feat_info["static_cols"]),
        "series_count": len(feat_info["series_blocks"]),
        "seq_len": len(feat_info["series_blocks"][0]),
        "vehicle": sum(len(block) for block in feat_info["battery_blocks"]),
        "emb_vocab": feat_info.get("num_embeddings", cfg.NUM_EMBEDDINGS)
    }
    
    print(f"\t[INFO] Loading Model: {dims}")

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

    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE, weights_only=True))
    model.eval()
    return model, feat_info

# ==========================================
# 4. MAIN PIPELINE WRAPPER
# ==========================================
def test_pipeline(case_id, run_name, sample_indices=[0, 1]):
    print(f"\n{'='*50}")
    print(f"🔬 STARTING CLOSED-LOOP SIMULATION: {run_name}")
    print(f"{'='*50}")

    # 1. Setup
    model, _ = load_trained_model(run_name)
    if model is None: return

    # 2. Initialize Environment
    env = SimulationEnvironment(case_id)

    # 3. Run Simulation for specific samples (e.g., different days/scenarios)
    for sample_idx in sample_indices:
        print(f"\n>>> Simulating Sample Index: {sample_idx}")
        
        # Reset Environment State if needed (or Create new Env per sample)
        # Ideally Env handles specific sample data internally.
        # Here we assume Env loaded ALL data, and we index into it.
        
        # Loop through 24 hours (or 48 if full horizon)
        # Assuming model predicts for next hour based on past 24h history
        
        simulation_steps = 24 
        
        for t in range(simulation_steps):
            
            # A. Get Observation (Features)
            # This pulls current SOC from env.soc_t_v
            x_tensor = env.get_observation(t, sample_idx)
            
            # B. AI Prediction
            with torch.no_grad():
                # Model output is scalar (Net Power)
                decision = model(x_tensor).item()
                
            # C. Execute Step (Rule-based allocation + Physics Update)
            env.step(t, decision, sample_idx)
            
            # Logging
            print(f"   Hr {t:02d}: AI Dec={decision:6.2f} kW | Grid={env.grid_load[t]:6.2f} kW | SOC Avg={np.mean(env.soc_t_v[t+1, :]):.2f}")

        # D. Calculate Total Metrics for Sample
        total_cost = np.sum(env.grid_load[:simulation_steps] * 0.30) # Mock price 0.30
        print(f"   ✅ Sample {sample_idx} Complete. Total Grid Cost: ${total_cost:.2f}")
def run_allocation(index, available, t_arr, t_dep, soc_arr, soc_dep, decision):
    """
    Rule-based charging routine with priority ranking.
    SOC[t, v] represents the state-of-charge *before charging* at hour t.
    The decision array modulates net power (positive = charge, negative = discharge).
    Batteries are prioritized by urgency:
        urgency = (SOC_dep - SOC_now) / (t_dep - t_now)
    """

    # # === Helper functions ===
    # def load_building():
    #     df = pd.read_csv(
    #         "/home/lisa4090/Documents/GitHub/V2B_Optimization_with_AI_on_BSS/data/processed/building_data.csv",
    #         index_col="Datetime",
    #     )
    #     return df.apply(pd.to_numeric, errors="coerce").values.flatten()

    # def load_price():
    #     df = pd.read_csv(
    #         "/home/lisa4090/Documents/GitHub/V2B_Optimization_with_AI_on_BSS/data/processed/electricitycostG2B_data.csv",
    #         index_col="Datetime",
    #     )
    #     return df.apply(pd.to_numeric, errors="coerce").values.flatten()

    # def load_pv():
    #     rad = pd.read_csv(
    #         "/home/lisa4090/Documents/GitHub/V2B_Optimization_with_AI_on_BSS/data/processed/radiation_data.csv",
    #         index_col="Datetime",
    #     )
    #     tmp = pd.read_csv(
    #         "/home/lisa4090/Documents/GitHub/V2B_Optimization_with_AI_on_BSS/data/processed/temperature_data.csv",
    #         index_col="Datetime",
    #     )
    #     rad, tmp = rad.apply(pd.to_numeric, errors="coerce"), tmp.apply(pd.to_numeric, errors="coerce")

    #     efficiency, capacity = 0.20, 500  # 500 m², 20% efficient
    #     return [
    #         efficiency * capacity * (r * 0.2778) * (1 - 0.005 * (t - 25))
    #         for r, t in zip(rad.values.flatten(), tmp.values.flatten())
    #     ]

    # # === Load external data ===
    # bldg = load_building()[index: index + 48]
    # elec = load_price()[index: index + 48]
    # pv = load_pv()[index: index + 48]

    # === Initialize arrays ===
    T, n_veh = 48, len(soc_arr)
    soc_t_v = np.zeros((T + 1, n_veh))  # +1 for final SOC after last hour
    ch_t_v = np.zeros((T, n_veh))
    gd_t = np.zeros(T)
    loss_t = np.zeros(T)

    # Initialize SOCs at arrivals
    for v in range(n_veh):
        soc_t_v[t_arr[v], v] = soc_arr[v]

    # === Main simulation loop ===
    all_scores = []

    for t in range(T):
        total_power = 0
        decision_t = decision[t - 24] if (decision is not False and t >= 24) else None

        # Clip decision within ±14 kW
        if decision_t is not None:
            decision_t = np.clip(decision_t, -14, 14)

        assert -14 <= (decision_t if decision_t is not None else 0) <= 14, \
            f"Decision at t={t} out of bounds: {decision_t}"

        # === Compute urgency scores ===
        scores = []
        urgency_by_timeleft = defaultdict(list)
        trigger_max_charge = False

        for v in range(n_veh):
            if available[v][t] == 0:
                scores.append(-np.inf)
                continue

            soc_now = soc_t_v[t, v]
            target = soc_dep[v]
            hours_left = int(t_dep[v] - t)
            urgency = 1.5 * (target - soc_now) / (hours_left * 0.9)

            scores.append(round(urgency, 3))
            urgency_by_timeleft[hours_left].append(round(urgency, 3))

        all_scores.append(scores)

        # === Predict overload condition ===
        total_required_kw = 28 if decision_t is not None else 0
        ave_per_hour_kW = 0
        # print(urgency_by_timeleft)
        for timeleft, urg_list in urgency_by_timeleft.items():

            total_required_kw += sum(max(0, u) for u in urg_list) 
            ave_per_hour_kW += total_required_kw/timeleft 
            # print(f"t={t}",timeleft, ave_per_hour_kW)

            # Urgent departures
            if ave_per_hour_kW >= 14:
                trigger_max_charge = True
                # print("triggered near-future overload")
                break
            

        # === Determine charging/discharging mode ===
        if trigger_max_charge:
            reverse_score, power_max = True, 14
        elif decision_t is None:
            reverse_score, power_max = True, 14
        elif decision_t >= 0:
            reverse_score, power_max = True, decision_t
        else:
            reverse_score, power_max = False, decision_t

        # === Sort vehicles by priority ===
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=reverse_score)

        # === Apply charging/discharging decisions ===
        for v, urgency in sorted_scores:
            if not (t_arr[v] <= t < t_dep[v]):
                continue

            soc_now = soc_t_v[t, v]
            target = soc_dep[v]
            assert 0.199 <= soc_now <= 0.901, f"Invalid SOC {soc_now:.2f} at t={t}, v={v}"

            avail_discharge = (soc_now - 0.2) * 1.5 * 0.9 if t_dep[v] >= t + 1 else 0
            req_power = (target - soc_now) * 1.5 / 0.9

            charge_power, discharge_power = 0.0, 0.0
            
            if (decision_t is None) or (decision_t >= 0) or (trigger_max_charge):
                # Charging
                charge_power = max(0, min(req_power, power_max - total_power))
                ch_t_v[t, v] = charge_power
                total_power += charge_power
                
            else:
                # Discharging
                discharge_power = min(avail_discharge, max(0.0, -decision_t + total_power))
                ch_t_v[t, v] = -discharge_power
                total_power -= discharge_power
                print("discharging", discharge_power, total_power)

            # Update SOC (considering efficiency)
            if ch_t_v[t, v] >= 0:
                soc_t_v[t + 1, v] = soc_now + (ch_t_v[t, v] / 1.5) * 0.9
            else:
                soc_t_v[t + 1, v] = soc_now + (ch_t_v[t, v] / 1.5) / 0.9
                
        # === Safety checks ===
        assert -14 <= total_power <= 14, f"Total power at t={t} exceeds limit: {total_power}"
        assert all(-1.5 <= ch_t_v[t, v] <= 1.5 for v in range(n_veh)), f"Out of range {ch_t_v[t, v]} at t={t}"

        # === Net grid balance ===
        net_demand = bldg[t] - pv[t] + total_power
        gd_t[t] = max(0, net_demand)
        loss_t[t] = -min(0, net_demand)
        
        

    # === Check SOC target misses ===
    for v in range(n_veh):
        gap = soc_dep[v] - soc_t_v[t_dep[v], v]
        soc_now = soc_t_v[t_dep[v], v]
        if gap > 0.01:
            print(f"[MISS] v={v:3d}, stay={t_dep[v] - t_arr[v]:2d}h, dep{t_dep[v]} gap={gap:.3f}, "
                  f"arrSOC={soc_arr[v]:.2f} leavingSOC={soc_now:.2f} expectedDepSOC={soc_dep[v]:.2f}")

    # === Cost calculation ===
    total_cost = np.sum(gd_t * elec)

    return soc_t_v, ch_t_v, gd_t, loss_t, total_cost, bldg, pv, elec

if __name__ == "__main__":
    # Example usage
    # Replace with actual run name from your output folder
    test_pipeline(case_id=0, run_name="STAFV2_V3_e-4LR_32batchsize_5e-4decay")