import math
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm


    def run_immediate_charging(self, index, available, t_arr, t_dep, soc_arr, soc_dep):
        """
        Simple heuristic charging routine — immediately charges batteries toward departure SOC
        within hourly power and efficiency limits. No optimization (rule-based control).
        """

        bldg = self.get_building_energy_demand()[index + 24: index + 24 + self.win_len]
        elec_G2B, elec_G2V = self.get_electricity_price()
        elec_G2B, elec_G2V = elec_G2B[index + 24: index + 24 + self.win_len], elec_G2V[index + 24: index + 24 + self.win_len]
        pv = self.get_photovoltaic_generation()[index + 24: index + 24 + self.win_len]

        n_veh = len(soc_arr)
        soc_t_v = np.zeros((self.win_len, n_veh))
        ch_t_v = np.zeros((self.win_len, n_veh))
        gd_t = np.zeros(self.win_len)
        loss_t = np.zeros(self.win_len)

        # initialize with arriving SOCs
        for i in range(n_veh):
            soc_t_v[t_arr[i],i] = soc_arr[i].copy()
            

        for t in range(self.win_len):
            total_charging_power = 0

            for v in range(n_veh):
                if t < t_arr[v] or t >= t_dep[v]:
                    continue  # not available

                # check if SOC < departure target
                if soc_t_v[t, v] < soc_dep[v] and available[v][t] == 1:
                    # how much SOC gap remains
                    delta_soc = soc_dep[v] - soc_t_v[t, v]

                    # power required to reach target
                    req_power = (delta_soc * self.batt_cap) / self.ch_eff

                    # enforce per-hour limit
                    charge_power = min(req_power, self.max_power - total_charging_power)
                    print(f"Time {t}, Vehicle {v}: Charging power set to {charge_power:.2f} kW")

                    if charge_power > 0:
                        ch_t_v[t, v] = charge_power
                        total_charging_power += charge_power

                # update SOC for next hour
                if t < self.win_len - 1:
                    soc_t_v[t + 1, v] = soc_t_v[t, v] + (ch_t_v[t, v] / self.batt_cap) * self.ch_eff
                    assert soc_t_v[t + 1, v] <= self.max_soc + 0.001, "Exceeded max SOC!"

            print("Charging profile:", np.array(ch_t_v).shape)
            # compute grid demand = building - pv + total charging
            gd_t[t] = max(0, bldg[t] - pv[t] + total_charging_power)
            loss_t[t] = -min(0, bldg[t] - pv[t] + total_charging_power)

        # compute total cost
        total_cost = np.sum([gd_t[t] * elec_G2B[t] for t in range(self.win_len)])

        print(f"Immediate charging run complete. Total cost = {total_cost:.2f}")

        return soc_t_v, ch_t_v, gd_t, loss_t, total_cost
    