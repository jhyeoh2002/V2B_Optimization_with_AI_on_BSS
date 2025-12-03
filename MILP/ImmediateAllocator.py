import math
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
from logstatus import log_status

import sys
import os
import numpy as np
from MILP.batteryreader import get_battery_details

sys.path.append(os.path.abspath(".."))

import config as cfg

class ImmediateCharging:

    def __init__(self, case_id):
        """
        Initialize the optimizer with configuration parameters.
        """
        self.batt_cap = cfg.BATTERY_CAPACITY  # Battery capacity in kWh
        self.max_power = cfg.CHARGING_RATE  # Maximum charging/discharging power in kW
        self.min_soc = cfg.MIN_SOC  # Minimum state of charge
        self.max_soc = cfg.MAX_SOC  # Maximum state of charge
        self.ch_eff = cfg.CHARGING_EFFICIENCY  # Charging efficiency
        self.win_len = cfg.WINDOW_LENGTH if case_id != 0 else 48 # Optimization window length

    def allocate(self, availability, t_dep, soc_arr, soc_dep, bldg, elec_G2B, elec_G2V, pv):
        """
        Simple heuristic charging routine — immediately charges batteries toward departure SOC
        within hourly power and efficiency limits. No optimization (rule-based control).
        """

        n_veh = len(soc_arr)
        
        gd_t = np.zeros(self.win_len)
        loss_t = np.zeros(self.win_len)
        G2B_t = np.zeros(self.win_len)
        P2B_t = np.zeros(self.win_len)
        
        G2V_t_v = np.zeros((self.win_len, n_veh))
        P2V_t_v = np.zeros((self.win_len, n_veh))
        V2B_t_v = np.zeros((self.win_len, n_veh))
        soc_t_v = np.zeros((self.win_len, n_veh))
        
        unmet_v = np.zeros(n_veh)

        soc_t_v[0, :] = soc_arr.copy()

        for t in range(self.win_len):
            total_charging_power = 0
            remaining_pv_t = pv[t]

            for v in range(n_veh):
                if availability[v][t] == 0: # vehicle not available
                    if t == 0:
                        soc_t_v[t,v] = soc_arr[v]
                    else:
                        soc_t_v[t,v] = soc_t_v[t-1,v]
                    continue  # not available

                # check if SOC < departure target
                elif soc_t_v[t, v] <= soc_dep[v] and availability[v][t] == 1:
                    
                    # how much SOC gap remains
                    if t == 0:
                        delta_soc = soc_dep[v] - soc_arr[v]
                    else:
                        delta_soc = soc_dep[v] - soc_t_v[t-1, v]

                    # print(f"Time {t}, Vehicle {v}: Current SOC {soc_t_v[t-1, v]:.3f}, Target SOC {soc_dep[v]:.3f}, Delta SOC {delta_soc:.3f}")

                    # power required to reach target
                    req_power = (delta_soc * self.batt_cap) / self.ch_eff

                    # enforce per-hour limit
                    charge_power = min(req_power, self.max_power - total_charging_power)
                    total_charging_power += charge_power

                    if remaining_pv_t > 0: #check PV availability
                        # use PV first
                        P2V_t_v[t, v] = min(charge_power, remaining_pv_t)
                        remaining_pv_t -= P2V_t_v[t, v]
                        
                        if charge_power - P2V_t_v[t, v] > 0: # if PV not enough, use grid
                            G2V_t_v[t, v] = charge_power - P2V_t_v[t, v]
                            
                    else: # no PV, use grid
                        G2V_t_v[t, v] = charge_power

                elif soc_t_v[t, v] < soc_dep[v] and t >= t_dep[v]:
                    # vehicle is available but past departure time, cannot charge
                    unmet_v[v] = (soc_dep[v] - soc_t_v[t, v]) * self.batt_cap
                    
                else:
                    raise ValueError(f"Unexpected condition for vehicle {v} at time {t}: SOC {soc_t_v[t, v]}, Departure time {t_dep[v]}, Availability {availability[v][t]}")
                    
                # update SOC for next hour
                if t == 0:
                    soc_t_v[t, v] = soc_arr[v] + ((G2V_t_v[t, v] + P2V_t_v[t, v]) / self.batt_cap) * self.ch_eff
                else:
                    soc_t_v[t, v] = soc_t_v[t - 1, v] + ((G2V_t_v[t, v] + P2V_t_v[t, v]) / self.batt_cap) * self.ch_eff

                assert soc_t_v[t, v] <= self.max_soc + 0.001, f"Exceeded max SOC for vehicle {v} at time {t}! soc: {soc_t_v[t, v]}"
                assert soc_t_v[t, v] >= self.min_soc - 0.001, f"Below min SOC for vehicle {v} at time {t}! soc: {soc_t_v[t, v]}"

            # store building flows
            P2B_t[t] = min(remaining_pv_t, bldg[t])
            remaining_pv_t -= P2B_t[t]
            
            G2B_t[t] = max(0, bldg[t] - P2B_t[t])

            # compute grid demand = building - pv + total charging
            gd_t[t] = G2B_t[t] + np.sum(G2V_t_v, axis=1)[t]
            
            loss_t[t] = remaining_pv_t # PV loss is any unused PV
            
        initial_demand = np.maximum(bldg - pv, 0)
        charging_demand = np.sum(G2V_t_v + P2V_t_v, axis=1)
        
        # compute total cost
        total_cost = np.sum([G2B_t[t] * elec_G2B[t] + (np.sum(G2V_t_v, axis=1)[t] * elec_G2V[t])for t in range(self.win_len)])

        total_grid_demand = np.sum(gd_t)
        total_demand = np.sum(G2B_t + P2B_t) + np.sum(G2V_t_v + P2V_t_v)
        pv_generated = np.sum(pv)
        pv_loss = np.sum(loss_t)

        self_sufficiency = (
            1 - (total_grid_demand / total_demand)
            if total_demand > 0 else 0
        )
        pv_utilization = (
            (pv_generated - pv_loss) / pv_generated
            if pv_generated > 0 else 0
        )
        
        # Structured return for downstream saving
        result = {
            "grid_demand": gd_t, # Grid demand over time (window length)
            "pv_loss": loss_t, # PV loss over time (window length)
            "initial_demand": initial_demand, # Initial demand (window length)
            "charging_demand": charging_demand, # Charging demand (window length)

            "G2B": G2B_t, # Grid to Building flows (window length)
            "P2B": P2B_t, # PV to Building flows (window length)

            "G2V": G2V_t_v, # Grid to Vehicle flows (n_vehicle x window length)
            "P2V": P2V_t_v, # PV to Vehicle flows (n_vehicle x window length)
            "V2B": V2B_t_v, # Vehicle to Building flows (n_vehicle x window length)

            "SOC": soc_t_v, # State of Charge over time (n_vehicle x window length)
            "unmet": unmet_v, # Unmet demand per vehicle (n_vehicle)
            
            "objective_value": None,
            "optimization_time_s": None,
            "mip_gap_percent": None,

            "total_cost": total_cost, # Total cost incurred (scalar)
            "self_sufficiency": self_sufficiency, # Self-sufficiency metric (scalar)
            "pv_utilization": pv_utilization, # PV utilization metric (scalar)
        }

        return result    