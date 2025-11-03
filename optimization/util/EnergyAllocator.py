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

    def __init__(self):
        """
        Initialize the EnergyAllocator with project configurations and data.
        """
        self.proj_name = cfg.PROJECT_NAME
        self.win_len = cfg.WINDOW_LENGTH
        self.output_dir = os.path.join(".", "output", self.proj_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data
        index, available, SOC_a_v, SOC_d_v, t_a_v, t_d_v = batteryreader.get_battery_details(self.win_len)
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

    def run_iterations(self, mode="optimization"):
        """
        Run iterations for either optimization or immediate charging.

        Parameters:
            mode (str): "optimization" or "immediate_charging".

        Returns:
            dict: Aggregated results for all iterations.
        """

        sys.stdout = open(os.path.join(self.output_dir, f"{mode}_log.txt"), "w")
        
        # Initialize result containers
        results = {
            "grid_demand": [], # Grid demand over time (window length)
            "pv_loss": [], # PV loss over time (window length)
            "initial_demand": [], # Initial demand (window length)
            "charging_demand": [], # Charging demand (window length)
            
            "G2B": [], # Grid to Building flows (window length)
            "P2B": [], # PV to Building flows (window length)
            
            "G2V": [], # Grid to Vehicle flows (n_vehicle x window length)
            "P2V": [], # PV to Vehicle flows (n_vehicle x window length)
            "V2B": [], # Vehicle to Building flows (n_vehicle x window length)

            "SOC": [], # State of Charge over time (n_vehicle x window length)
            "unmet": [], # Unmet demand per vehicle (n_vehicle)

            "objective_value":[], # Final objective value (scalar)
            "optimization_time_s": [], # Optimization time in seconds (scalar)
            "mip_gap_percent": [], # MIP gap percentage (scalar)
            "total_cost": [], # Total cost incurred (scalar)
            "self_sufficiency": [], # Self-sufficiency metric (scalar)
            "pv_utilization": [], # PV utilization metric (scalar)
        }

        for n in tqdm(range(self.n_list), desc=f"Processing {mode} series", unit="series"):
            # Extract current data for the iteration
            index_current = self.window_index[n]
            availability_current = np.array(self.available[n])
            SOC_a_v_current = np.array(self.SOC_a_v[n])
            SOC_d_v_current = np.array(self.SOC_d_v[n])
            t_d_v_current = np.array(self.t_d_v[n])

            bldg_current = self.bldg[index_current : index_current + self.win_len]
            elec_G2B_current = self.elec_G2B[index_current : index_current + self.win_len]
            elec_G2V_current = self.elec_G2V[index_current : index_current + self.win_len]
            pv_current = self.pv[index_current : index_current + self.win_len]

            print("=" * 60)
            print(f"Starting series num: {n}, Datetime Index: {index_current}")
            print("-" * 60)

            # Run the appropriate method based on the mode
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
                    pv=pv_current,
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
                    pv=pv_current,
                )
                
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Append results
            if result:
                for key in results:
                    results[key].append(result[key])
            
        for key, value in results.items():
            print(f"\n\nSaving {key} results...\n")
            
            if key in ["unmet", "G2V", "P2V", "V2B", "SOC"]:
                value = self.pad_vehicle_dimension(value)
                
            else:
                value = np.array(value)

            print(value)
            print(f"\nShape: {np.array(value).shape}\n")
            results[key] = value
            
            self.save_results(key, value, mode)

        return results

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
