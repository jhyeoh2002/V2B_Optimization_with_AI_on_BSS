import sys
import numpy as np
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config as cfg
import batteryreader
from GurobiOptimizer import GurobiOptimizer
from ImmediateAllocator import ImmediateCharging
from logstatus import log_status
from tqdm import tqdm
from readdata import DataReader

class EnergyAllocator:

    def __init__(self, case_id=1, tolerance=25):
        """
        Initialize the EnergyAllocator for a specific Case ID.
        """
        self.proj_name = cfg.PROJECT_NAME
        self.win_len = cfg.WINDOW_LENGTH if case_id != 0 else 48
        self.tolerance = tolerance
        self.case_id = case_id
        
        # --- PATH CONFIGURATION ---
        # 1. Define Input Directory (Where data comes from)
        base_data_dir = f"{cfg.BATTERYDEMAND_DIR}/"
        case_map = {
            0: "case0_test",
            1: "case1_real_only",
            2: "case2_nan_filled",
            3: "case3_extended_generated"
        }
        case_name = case_map.get(case_id, f"case{case_id}")
        self.input_case_dir = os.path.join(base_data_dir, case_name)

        # 2. Define Output Directory (Where results go)
        # Result path: ./data/optimization_results/tol25/case1_real_only/
        self.output_dir = os.path.join(
            cfg.OPTRESULTS_DIR, 
            case_name
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # --- LOAD DATA ---
        # Pass the specific input directory to the reader
        index, available, SOC_a_v, SOC_d_v, t_a_v, t_d_v = batteryreader.get_battery_details(self.input_case_dir)
        
        self.bldg = DataReader.get_building_energy_demand()
        self.elec_G2B, self.elec_G2V = DataReader.get_electricity_price()
        self.pv = DataReader.get_photovoltaic_generation()

        # Store battery details
        self.window_index = index
        self.available = available
        self.SOC_a_v = SOC_a_v
        self.SOC_d_v = SOC_d_v
        self.t_a_v = t_a_v
        self.t_d_v = t_d_v
        self.n_list = len(index)
        self.max_vehicles = max(len(list) for list in SOC_a_v)
             
        
    def run_iterations(self, mode="optimization", rerun=False, checkpoint_interval=10):
        """
        Run iterations for either optimization or immediate charging, with crash-safe checkpointing.
        
        Parameters:
            mode (str): "optimization" or "immediate_charging".
            rerun (bool): If True, ignore any existing checkpoint and start fresh.
            checkpoint_interval (int): Save a checkpoint every N iterations.
        
        Returns:
            dict: Aggregated results for all iterations.
        """

        # Setup paths
        checkpoint_base = os.path.join(self.output_dir, f"{mode}_checkpoint")
        checkpoint_file = checkpoint_base + ".npz"
        tmp_file        = checkpoint_base + ".tmp.npz"
        result_file =  os.path.join(self.output_dir, mode, "V2B.npy")
        
        if os.path.exists(result_file) and not rerun:
            print(f"\t\t[INFO] Results already exist at '{result_file}'. Skipping {mode} run.")
            
            # if mode == "optimization":
            #     plot_energy_flows_for_random_scenario(
            #             case=self.case_id,
            #             RESULTS_PATH_OPT=os.path.join(self.output_dir, "optimization"),
            #             RESULTS_PATH_IMM=os.path.join(self.output_dir, "immediate_charging")
            # )
                    
            return

        # Redirect stdout to log for this mode
        # sys.stdout = open(os.path.join(self.output_dir, f"{mode}_log.txt"), "w")

        # Prepare result containers
        result_keys = [
            "grid_demand","pv_loss","initial_demand","charging_demand",
            "G2B","P2B","G2V","P2V","V2B","SOC","unmet","objective_value",
            "optimization_time_s","mip_gap_percent","total_cost",
            "self_sufficiency","pv_utilization"
        ]
        results = { key: [] for key in result_keys }

        # Determine starting iteration
        start_n = 0
           
        if (not rerun) and os.path.exists(checkpoint_file):
            print(f"\t\t[INFO] Found checkpoint: '{checkpoint_file}'")
            ckpt = np.load(checkpoint_file, allow_pickle=True)
            
            # Load partial results back into memory
            for key in result_keys:
                if key in ckpt:
                    results[key] = ckpt[key].tolist()
                else:
                    results[key] = []
            
            last_n = int(ckpt["last_n"])
            start_n = last_n + 1
            print(f"\t\t[INFO] Resuming from iteration {start_n}")
        else:
            print(f"\t\t[INFO] Starting fresh run: mode={mode}")
            
        np.random.seed(42)  # For reproducibility
        max_index = len(self.bldg) - self.win_len
        extended_index = np.random.randint(0, max_index, size=len(self.window_index))
        
        # Main loop
        for n in tqdm(range(start_n, self.n_list), desc=f"\t\t[INFO] Processing {mode} series", unit="series", ncols=100):
            index_current        = self.window_index[n] if self.window_index[n] >=0 else extended_index[n]
            availability_current = np.array(self.available[n])
            SOC_a_v_current      = np.array(self.SOC_a_v[n])
            SOC_d_v_current      = np.array(self.SOC_d_v[n])
            t_d_v_current        = np.array(self.t_d_v[n])
            
            bldg_current    = self.bldg[index_current : index_current + self.win_len]
            elec_G2B_current = self.elec_G2B[index_current : index_current + self.win_len]
            elec_G2V_current = self.elec_G2V[index_current : index_current + self.win_len]
            pv_current      = self.pv[index_current : index_current + self.win_len]

            # print("="*60)
            # print(f"Starting series num: {n}, Datetime Index: {index_current}")
            # print("-"*60)

            if mode == "optimization":
                optimizer = GurobiOptimizer(CASE_ID=self.case_id)
                result = optimizer.optimize(
                    availability=availability_current,
                    t_dep=t_d_v_current,
                    soc_arr=SOC_a_v_current,
                    soc_dep=SOC_d_v_current,
                    bldg=bldg_current,
                    elec_G2B=elec_G2B_current,
                    elec_G2V=elec_G2V_current,
                    pv=pv_current
                )
            elif mode == "immediate_charging":
                allocator = ImmediateCharging(case_id=self.case_id)
                result = allocator.allocate(
                    availability=availability_current,
                    t_dep=t_d_v_current,
                    soc_arr=SOC_a_v_current,
                    soc_dep=SOC_d_v_current,
                    bldg=bldg_current,
                    elec_G2B=elec_G2B_current,
                    elec_G2V=elec_G2V_current,
                    pv=pv_current
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            if result:
                for key in results:
                    results[key].append(result[key])
                    
                    # if key == "objective_value":
                        # print(f"length of optimization values: {len(results[key])}")
            
            # --- 5. Checkpoint every few iterations ---
            if ((n + 1) % checkpoint_interval == 0) or (n == self.n_list - 1):
                try:
                    # Pad irregular arrays using your internal function
                    save_dict = {}
                    for k, v in results.items():
                        if k in ["unmet", "G2V", "P2V", "V2B", "SOC"]:
                            padded_v = self.pad_vehicle_dimension(v)
                            save_dict[k] = padded_v
                        else:
                            save_dict[k] = np.array(v)
                    save_dict["last_n"] = n

                    np.savez_compressed(tmp_file, **save_dict)
                    os.replace(tmp_file, checkpoint_file)
                    # print(f"\t\t[CHECKPOINT] Saved checkpoint at iteration {n}")
                except Exception as e:
                    print(f"\t\t[ERROR] Checkpoint save failed at iteration {n}: {e}")

        # After loop: convert & save full results
        for key, value in results.items():
            print(f"\t\t[INFO] Saving {key} results...")
            if key in ["unmet", "G2V", "P2V", "V2B", "SOC"]:
                value = self.pad_vehicle_dimension(value)
            else:
                value = np.array(value)
            results[key] = value
            self.save_results(key, value, mode)
            

        if mode == "optimization":
            plot_energy_flows_for_random_scenario(
                    case=self.case_id,
                    RESULTS_PATH_OPT=os.path.join(self.output_dir, "optimization"),
                    RESULTS_PATH_IMM=os.path.join(self.output_dir, "immediate_charging")
        )
            
        # Clean up checkpoint
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                print("\t\t[INFO] Deleted checkpoint after full completion.")
            except Exception as e:
                print(f"\t\t[WARN] Could not delete checkpoint file: {e}")
                
        return 


    def save_results(self, key, value, mode):
        """
        Save results to .npy files.

        Parameters:
            results (dict): Aggregated results to save.
            mode (str): "optimization" or "immediate_charging".
        """
        output_dir = os.path.join(self.output_dir, mode)
        os.makedirs(output_dir, exist_ok=True)
        
        print((f"\t\t[INFO] Saving results for key: {key} to {output_dir} with shape {value.shape}"))

        np.save(os.path.join(output_dir, f"{key}.npy"), np.array(value))
            
            
    def pad_vehicle_dimension(self, array_list, fill_value=np.nan):
        """
        Pads a list of arrays so that all have the same number of vehicles (n_veh).
        Works for:
            - 1D arrays: (n_veh,)  -> per-vehicle static values (e.g., unmet)
            - 2D arrays: (window_len, n_veh)
        
        Skips arrays shaped (window_len,), which are per-time-step only (no vehicle dimension).
        
        Parameters
        ----------
        array_list : list of np.ndarray
            Each element is (n_veh,) or (window_len, n_veh)
        fill_value : float, optional
            Value used for padding (default: np.nan)
        
        Returns
        -------
        stacked : np.ndarray
            Stacked array with shape:
                - (n_runs, max_n_veh) for (n_veh,)
                - (n_runs, window_len, max_n_veh) for (window_len, n_veh)
        """
        padded_list = []
        
        for arr in array_list:
            arr = np.asarray(arr, dtype=float)

            # (window_len,) time series → skip this function entirely in your main loop
            if arr.ndim == 1 and arr.shape[0] == self.win_len:
                padded = arr  # No padding needed
            
            # Case 1: unmet (n_veh,)
            elif arr.ndim == 1:
                pad_width = (0, self.max_vehicles - arr.shape[0])
                padded = np.pad(arr, pad_width, constant_values=fill_value)
            
            # Case 2: (window_len, n_veh)
            elif arr.ndim == 2:
                pad_width = ((0, 0), (0, self.max_vehicles - arr.shape[1]))
                padded = np.pad(arr, pad_width, constant_values=fill_value)
            
            else:
                raise ValueError(f"Unexpected array shape {arr.shape}")

            padded_list.append(padded)

        # Stack results
        first_arr = padded_list[0]
        if first_arr.ndim == 1 or first_arr.ndim == 2:
            stacked = np.stack(padded_list, axis=0)  # (n_runs, max_n_veh)
        else:
            raise ValueError(f"Unexpected padded array shape {first_arr.shape}")

        return stacked
    
import numpy as np
import matplotlib.pyplot as plt
import random

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_energy_flows_for_random_scenario(
    case: int,
    RESULTS_PATH_OPT,
    RESULTS_PATH_IMM
):
    """
    Plot stacked energy-flow diagrams comparing Immediate vs Optimized charging,
    including statistical text annotations.
    """

    SAVE_DIR = f"./figures/optimization{case}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------
    # Optimization Results
    G2B_OPT = np.load(f"{RESULTS_PATH_OPT}/G2B.npy")
    G2V_OPT = np.load(f"{RESULTS_PATH_OPT}/G2V.npy")
    P2B_OPT = np.load(f"{RESULTS_PATH_OPT}/P2B.npy")
    P2V_OPT = np.load(f"{RESULTS_PATH_OPT}/P2V.npy")
    V2B_OPT = np.load(f"{RESULTS_PATH_OPT}/V2B.npy")
    pv_loss_opt = np.load(f"{RESULTS_PATH_OPT}/pv_loss.npy")
    grid_demand_opt = np.load(f"{RESULTS_PATH_OPT}/grid_demand.npy")
    # Metrics
    total_cost_opt = np.load(f"{RESULTS_PATH_OPT}/total_cost.npy")
    self_suff_opt = np.load(f"{RESULTS_PATH_OPT}/self_sufficiency.npy")
    pv_util_opt = np.load(f"{RESULTS_PATH_OPT}/pv_utilization.npy")

    # Immediate Results
    G2B_IMM = np.load(f"{RESULTS_PATH_IMM}/G2B.npy")
    G2V_IMM = np.load(f"{RESULTS_PATH_IMM}/G2V.npy")
    P2B_IMM = np.load(f"{RESULTS_PATH_IMM}/P2B.npy")
    P2V_IMM = np.load(f"{RESULTS_PATH_IMM}/P2V.npy")
    V2B_IMM = np.load(f"{RESULTS_PATH_IMM}/V2B.npy")
    pv_loss_imm = np.load(f"{RESULTS_PATH_IMM}/pv_loss.npy")
    grid_demand_imm = np.load(f"{RESULTS_PATH_IMM}/grid_demand.npy")
    # Metrics
    total_cost_imm = np.load(f"{RESULTS_PATH_IMM}/total_cost.npy")
    self_suff_imm = np.load(f"{RESULTS_PATH_IMM}/self_sufficiency.npy")
    pv_util_imm = np.load(f"{RESULTS_PATH_IMM}/pv_utilization.npy")

    N = len(G2B_OPT)

    # -------------------------------------------------------------
    # 2. Helper functions
    # -------------------------------------------------------------
    def extend_last(ts):
        """Extend timeseries by 1 duplicate for step plotting."""
        ts = np.asarray(ts).flatten()
        return np.append(ts, ts[-1])

    def collapse_vehicle_flows(arr):
        """Sum across valid vehicle columns."""
        mat = arr[:, ~np.isnan(arr).any(axis=0)]
        return np.sum(mat, axis=1)

    # -------------------------------------------------------------
    # 3. Main plotting loop
    # -------------------------------------------------------------
    for idx in tqdm(range(N)):

        # --- Prepare Data Arrays ---
        G2B_i = extend_last(G2B_IMM[idx])
        G2V_i = extend_last(collapse_vehicle_flows(G2V_IMM[idx]))
        P2B_i = extend_last(P2B_IMM[idx])
        P2V_i = extend_last(collapse_vehicle_flows(P2V_IMM[idx]))
        V2B_i = extend_last(collapse_vehicle_flows(V2B_IMM[idx]))
        LOSS_i = extend_last(pv_loss_imm[idx])
        GRID_i = extend_last(grid_demand_imm[idx])

        G2B_o = extend_last(G2B_OPT[idx])
        G2V_o = extend_last(collapse_vehicle_flows(G2V_OPT[idx]))
        P2B_o = extend_last(P2B_OPT[idx])
        P2V_o = extend_last(collapse_vehicle_flows(P2V_OPT[idx]))
        V2B_o = extend_last(collapse_vehicle_flows(V2B_OPT[idx]))
        LOSS_o = extend_last(pv_loss_opt[idx])
        GRID_o = extend_last(grid_demand_opt[idx])

        # --- Calculate Metrics & Percentages for this index ---
        cost_i = float(total_cost_imm[idx])
        cost_o = float(total_cost_opt[idx])
        suff_i = float(self_suff_imm[idx])
        suff_o = float(self_suff_opt[idx])
        util_i = float(pv_util_imm[idx])
        util_o = float(pv_util_opt[idx])

        # Handle division by zero if necessary, though unlikely for cost
        cost_pct = (cost_o - cost_i) / cost_i * 100 if cost_i != 0 else 0
        suff_pct = (suff_o - suff_i) / suff_i * 100 if suff_i != 0 else 0
        util_pct = (util_o - util_i) / util_i * 100 if util_i != 0 else 0

        # --- Figure Setup ---
        # Time axis
        T = np.arange(len(G2B_i) - 1)
        T_ext = np.append(T, T[-1] + 1)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=300, sharex=True)

        colors = ["#e8e8f0", "#b2a2d4", "#2171b5", "#31a354", "#b3d1ac", "#808080"]
        
        scenarios = [
            ("Immediate Charging", G2B_i, G2V_i, V2B_i, P2B_i, P2V_i, LOSS_i, GRID_i),
            ("Optimized Charging", G2B_o, G2V_o, V2B_o, P2B_o, P2V_o, LOSS_o, GRID_o),
        ]

        # Calculate Y-limits dynamically
        ymax = max(
            np.max(G2B_i + G2V_i + V2B_i + P2B_i + P2V_i),
            np.max(G2B_o + G2V_o + V2B_o + P2B_o + P2V_o)
        ) * 1.5  # Extra space for text box
        ymin = min(-np.max(LOSS_i), -np.max(LOSS_o)) * 1.3

        for ax, (title, G2B, G2V, V2B, P2B, P2V, LOSS, GRID) in zip(axes, scenarios):
            cumulative = np.zeros_like(T_ext, dtype=float)

            # Grid Demand Line
            ax.step(T_ext, GRID, where='post', color='black', linewidth=1.5, label="Final Grid Demand")

            # PV Loss
            ax.fill_between(T_ext, cumulative, cumulative - LOSS, step='post',
                            color=colors[5], alpha=0.5, label="PV Loss")

            # G2B
            ax.fill_between(T_ext, cumulative, cumulative + G2B, step='post',
                            color=colors[0], label="Grid→Building")
            cumulative += G2B

            # G2V
            ax.fill_between(T_ext, cumulative, cumulative + G2V, step='post',
                            color=colors[1], alpha=0.5, label="Grid→Battery")
            cumulative += G2V

            # V2B
            ax.fill_between(T_ext, GRID, GRID + V2B, step='post',
                            color=colors[2], alpha=0.5, hatch="//", label="V2B")

            # PV to Building
            ax.fill_between(T_ext, GRID + V2B, GRID + V2B + P2B, step='post',
                            color=colors[3], alpha=0.5, hatch="//", label="PV→Building")

            # PV to Battery
            ax.fill_between(T_ext, GRID + V2B + P2B, GRID + V2B + P2B + P2V, step='post',
                            color=colors[4], alpha=0.5, hatch="//", label="PV→Battery")

            ax.set_title(title, fontweight="bold")
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(0, len(T))
            ax.set_ylabel("Power (kW)")
            ax.set_xticks(T[::3])
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)

            # ---------------------------------------------------------
            # NEW: Add Text Statistics
            # ---------------------------------------------------------
            # We use transform=ax.transAxes so (0.98, 0.95) is always 
            # top-right, regardless of x-axis scale.
            
            if "Immediate" in title:
                stats_text = (
                    r"$\mathbf{Total\ Cost\ ↓:}$"      + f"  NTD {cost_i:,.2f}\n"
                    r"$\mathbf{Self\text{-}Sufficiency\ ↑:}$" + f"  {suff_i*100:.1f}%\n"
                    r"$\mathbf{PV\ Utilization\ ↑:}$"  + f"  {util_i*100:.1f}%"
                )
            else:
                stats_text = (
                    r"$\mathbf{Total\ Cost\ ↓:}$"      + f"  NTD {cost_o:,.2f} ({cost_pct:+.1f}%)\n"
                    r"$\mathbf{Self\text{-}Sufficiency\ ↑:}$" + f"  {suff_o*100:.1f}%  ({suff_pct:+.1f}%)\n"
                    r"$\mathbf{PV\ Utilization\ ↑:}$"  + f"  {util_o*100:.1f}%  ({util_pct:+.1f}%)"
                )

            ax.text(
                0.98, 0.95, 
                stats_text,
                fontsize=11,
                ha='right', va='top',
                transform=ax.transAxes, # Relative coordinates (0-1)
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgrey'),
                linespacing=1.3
            )

        axes[-1].set_xlabel("Time Step")

        # ---------------------------------------------------------
        # Legend & Layout
        # ---------------------------------------------------------
        handles, labels = [], []
        seen_labels = set()
        for ax in axes:
            h_list, l_list = ax.get_legend_handles_labels()
            for h, l in zip(h_list, l_list):
                if l not in seen_labels:
                    handles.append(h)
                    labels.append(l)
                    seen_labels.add(l)

        plt.tight_layout()
        plt.subplots_adjust(right=0.82) 
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.84, 0.5), frameon=False)

        plt.savefig(f"{SAVE_DIR}/scenario_{idx}.png", dpi=300)
        plt.close()
        
# def plot_energy_flows_for_random_scenario(
#     case: int,
#     RESULTS_PATH_OPT,
#     RESULTS_PATH_IMM
# ):
#     """
#     Plot stacked energy-flow diagrams comparing:
#     (1) Immediate charging
#     (2) Optimized bi-directional charging

#     A random scenario index will be selected.
#     """


#     SAVE_DIR = f"./figures/optimization{case}"
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # -------------------------------------------------------------
#     # Load data
#     # -------------------------------------------------------------
#     G2B_OPT = np.load(f"{RESULTS_PATH_OPT}/G2B.npy")
#     G2V_OPT = np.load(f"{RESULTS_PATH_OPT}/G2V.npy")
#     P2B_OPT = np.load(f"{RESULTS_PATH_OPT}/P2B.npy")
#     P2V_OPT = np.load(f"{RESULTS_PATH_OPT}/P2V.npy")
#     V2B_OPT = np.load(f"{RESULTS_PATH_OPT}/V2B.npy")
#     pv_loss_opt = np.load(f"{RESULTS_PATH_OPT}/pv_loss.npy")
#     grid_demand_opt = np.load(f"{RESULTS_PATH_OPT}/grid_demand.npy")
#     total_cost_opt = np.load(f"{RESULTS_PATH_OPT}/total_cost.npy")
#     self_suff_opt = np.load(f"{RESULTS_PATH_OPT}/self_sufficiency.npy")
#     pv_util_opt = np.load(f"{RESULTS_PATH_OPT}/pv_utilization.npy")

#     G2B_IMM = np.load(f"{RESULTS_PATH_IMM}/G2B.npy")
#     G2V_IMM = np.load(f"{RESULTS_PATH_IMM}/G2V.npy")
#     P2B_IMM = np.load(f"{RESULTS_PATH_IMM}/P2B.npy")
#     P2V_IMM = np.load(f"{RESULTS_PATH_IMM}/P2V.npy")
#     V2B_IMM = np.load(f"{RESULTS_PATH_IMM}/V2B.npy")
#     pv_loss_imm = np.load(f"{RESULTS_PATH_IMM}/pv_loss.npy")
#     grid_demand_imm = np.load(f"{RESULTS_PATH_IMM}/grid_demand.npy")
#     total_cost_imm = np.load(f"{RESULTS_PATH_IMM}/total_cost.npy")
#     self_suff_imm = np.load(f"{RESULTS_PATH_IMM}/self_sufficiency.npy")
#     pv_util_imm = np.load(f"{RESULTS_PATH_IMM}/pv_utilization.npy")

#     N = len(G2B_OPT)


#     # -------------------------------------------------------------
#     # Helper functions
#     # -------------------------------------------------------------
#     def extend_last(ts):
#         """Extend timeseries by 1 duplicate for step plotting."""
#         ts = np.asarray(ts).flatten()
#         return np.append(ts, ts[-1])


#     def collapse_vehicle_flows(arr):
#         """Sum across valid vehicle columns."""
#         mat = arr[:, ~np.isnan(arr).any(axis=0)]
#         return np.sum(mat, axis=1)


#     # -------------------------------------------------------------
#     # Main plotting loop
#     # -------------------------------------------------------------
#     for idx in range(N):

#         # Prepare IMM arrays
#         G2B_i = extend_last(G2B_IMM[idx])
#         G2V_i = extend_last(collapse_vehicle_flows(G2V_IMM[idx]))
#         P2B_i = extend_last(P2B_IMM[idx])
#         P2V_i = extend_last(collapse_vehicle_flows(P2V_IMM[idx]))
#         V2B_i = extend_last(collapse_vehicle_flows(V2B_IMM[idx]))
#         LOSS_i = extend_last(pv_loss_imm[idx])
#         GRID_i = extend_last(grid_demand_imm[idx])

#         # Prepare OPT arrays
#         G2B_o = extend_last(G2B_OPT[idx])
#         G2V_o = extend_last(collapse_vehicle_flows(G2V_OPT[idx]))
#         P2B_o = extend_last(P2B_OPT[idx])
#         P2V_o = extend_last(collapse_vehicle_flows(P2V_OPT[idx]))
#         V2B_o = extend_last(collapse_vehicle_flows(V2B_OPT[idx]))
#         LOSS_o = extend_last(pv_loss_opt[idx])
#         GRID_o = extend_last(grid_demand_opt[idx])

#         # Metrics
#         cost_i = float(total_cost_imm[idx])
#         cost_o = float(total_cost_opt[idx])
#         suff_i = float(self_suff_imm[idx])
#         suff_o = float(self_suff_opt[idx])
#         util_i = float(pv_util_imm[idx])
#         util_o = float(pv_util_opt[idx])

#         cost_pct = (cost_o - cost_i) / cost_i * 100
#         suff_pct = (suff_o - suff_i) / suff_i * 100
#         util_pct = (util_o - util_i) / util_i * 100

#         # Time axis
#         T = np.arange(len(G2B_i) - 1)
#         T_ext = np.append(T, T[-1] + 1)

#         # ---------------------------------------------------------
#         # Begin figure
#         # ---------------------------------------------------------
#         fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=300, sharex=True)

#         colors = ["#e8e8f0", "#b2a2d4", "#2171b5", "#31a354", "#b3d1ac", "#808080"]
#         hatches = [None, None, "//", "//", "//", None]
#         labels = ["G→B", "G→Batt", "V2B", "PV→Bldg", "PV→Batt", "PV Loss"]

#         scenarios = [
#             ("Immediate Charging", G2B_i, G2V_i, V2B_i, P2B_i, P2V_i, LOSS_i, GRID_i),
#             ("Optimized Charging", G2B_o, G2V_o, V2B_o, P2B_o, P2V_o, LOSS_o, GRID_o),
#         ]

#         ymax = max(
#             np.max(G2B_i + G2V_i + V2B_i + P2B_i + P2V_i),
#             np.max(G2B_o + G2V_o + V2B_o + P2B_o + P2V_o)
#         ) * 1.4
#         ymin = min(-np.max(LOSS_i), -np.max(LOSS_o)) * 1.3

#         for ax, (title, G2B, G2V, V2B, P2B, P2V, LOSS, GRID) in zip(axes, scenarios):
#             cumulative = np.zeros_like(T_ext, dtype=float)

#             ax.step(T_ext, GRID, where='post', color='black', linewidth=1.5)

#             # PV Loss
#             ax.fill_between(T_ext, cumulative, cumulative - LOSS, step='post',
#                             color=colors[5], alpha=0.5, label="PV Loss")

#             # G2B
#             ax.fill_between(T_ext, cumulative, cumulative + G2B, step='post',
#                             color=colors[0], label="Grid→Building")
#             cumulative += G2B

#             # G2V
#             ax.fill_between(T_ext, cumulative, cumulative + G2V, step='post',
#                             color=colors[1], alpha=0.5, label="Grid→Battery")
#             cumulative += G2V

#             # V2B
#             ax.fill_between(T_ext, GRID, GRID + V2B, step='post',
#                             color=colors[2], alpha=0.5, hatch="//", label="V2B")

#             # PV to Building
#             ax.fill_between(T_ext, GRID + V2B, GRID + V2B + P2B, step='post',
#                             color=colors[3], alpha=0.5, hatch="//", label="PV→Building")

#             # PV to Battery
#             ax.fill_between(T_ext, GRID + V2B + P2B, GRID + V2B + P2B + P2V, step='post',
#                             color=colors[4], alpha=0.5, hatch="//", label="PV→Battery")

#             ax.set_title(title, fontweight="bold")
#             ax.set_ylim(ymin, ymax)
#             ax.set_xlim(0, len(T))
#             ax.set_ylabel("Power (kW)")
#             ax.set_xticks(T[::3])
            
#             ax.grid(True, axis='y', linestyle='--', alpha=0.5)

#         axes[-1].set_xlabel("Time Step")

       
#         # ---------------------------------------------------------
#         # FIX: Robust Legend Generation
#         # ---------------------------------------------------------
        
#         # 1. Collect handles/labels from BOTH axes (deduplicated)
#         #    This ensures if "V2B" is missing in one plot but present in the other, it still shows.
#         handles, labels = [], []
#         seen_labels = set()
        
#         for ax in axes:
#             h_list, l_list = ax.get_legend_handles_labels()
#             for h, l in zip(h_list, l_list):
#                 if l not in seen_labels:
#                     handles.append(h)
#                     labels.append(l)
#                     seen_labels.add(l)

#         # 2. Tight Layout FIRST
#         plt.tight_layout()

#         # 3. Adjust right margin to make room for legend
#         plt.subplots_adjust(right=0.82) 

#         # 4. Place legend in the reserved space
#         fig.legend(handles, labels, loc='center left', 
#                    bbox_to_anchor=(0.84, 0.5), 
#                    frameon=False)

#         # Save
#         plt.savefig(f"{SAVE_DIR}/scenario_{idx}.png", dpi=300)
#         plt.close()