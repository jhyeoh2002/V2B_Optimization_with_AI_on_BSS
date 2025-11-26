import sys
from tabnanny import check
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
        self.win_len = cfg.WINDOW_LENGTH
        self.tolerance = tolerance
        self.case_id = case_id
        
        # --- PATH CONFIGURATION ---
        # 1. Define Input Directory (Where data comes from)
        base_data_dir = f"./data/battery_demand/tol{self.tolerance}/"
        case_map = {
            1: "case1_real_only",
            2: "case2_nan_filled",
            3: "case3_extended_generated"
        }
        case_name = case_map.get(case_id, f"case{case_id}")
        self.input_case_dir = os.path.join(base_data_dir, case_name)

        # 2. Define Output Directory (Where results go)
        # Result path: ./data/optimization_results/tol25/case1_real_only/
        self.output_dir = os.path.join(
            "./data/optimization_results", 
            f"tol{self.tolerance}", 
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
        extended_index = np.random.uniform(low=0, high=15287, size=len(self.window_index)).astype(int)

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
                optimizer = GurobiOptimizer()
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
                allocator = ImmediateCharging()
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
