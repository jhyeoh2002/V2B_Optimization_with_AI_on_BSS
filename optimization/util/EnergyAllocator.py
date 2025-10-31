import sys
import numpy as np
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config as cfg
import batteryreader
from GurobiOptimizer import GurobiOptimizer
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

    def run_iterations(self, mode="optimization"):
        """
        Run iterations for either optimization or immediate charging.

        Parameters:
            mode (str): "optimization" or "immediate_charging".

        Returns:
            dict: Aggregated results for all iterations.
        """

        os.stdout = open(os.path.join(self.output_dir, f"{mode}_log.txt"), "w")
        
        # Initialize result containers
        results = {
            "grid_demand": [],
            "pv_loss": [],
            "G2B": [],
            "P2B": [],
            "G2V": [],
            "P2V": [],
            "V2B": [],
            "SOC": [],
            "unmet": [],
            "summary": [],
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
                result = self.run_immediate_charging(
                    index=index_current,
                    availability=availability_current,
                    t_a_v=self.t_a_v[n],
                    t_d_v=t_d_v_current,
                    SOC_a_v=SOC_a_v_current,
                    SOC_d_v=SOC_d_v_current,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Append results
            if result:
                for key in results:
                    results[key].append(result[key])
                    
            if n == 3:
                break

        self.save_results(results, mode)

        return results

    def save_results(self, results, mode):
        """
        Save results to .npy files.

        Parameters:
            results (dict): Aggregated results to save.
            mode (str): "optimization" or "immediate_charging".
        """
        output_dir = os.path.join(self.output_dir, mode)
        os.makedirs(output_dir, exist_ok=True)

        for key, value in results.items():
            np.save(os.path.join(output_dir, f"{key}.npy"), np.array(value))
