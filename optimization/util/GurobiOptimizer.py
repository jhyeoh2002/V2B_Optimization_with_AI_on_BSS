import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time
import sys
import os
import pandas as pd
import numpy as np
import sys
import gc, time

sys.path.append(os.path.abspath(".."))

from util.config import PROJECT_NAME, PV_AREA, WINDOW_LENGTH, BATTERY_CAPACITY, CHARGING_RATE, CHARGING_EFFICIENCY, MIN_SOC, MAX_SOC

# ==============================
# VEHICLE-TO-BUILDING OPTIMIZATION FUNCTION
# ==============================

class GurobiOptimizer:
    """
    A class to perform Vehicle-to-Building (V2B) optimization.

    Attributes:
    -----------
    batt_cap : int
        Battery capacity of the vehicles (kWh).
    max_power : float
        Maximum charging/discharging power (kW).
    min_soc : float
        Minimum allowable state of charge (SOC).
    max_soc : float
        Maximum allowable state of charge (SOC).
    ch_eff : float
        Charging efficiency.
    win_len : int
        Number of time steps in the optimization window.
    gamma_peak : float
        Weight for peak shaving objective.
    gamma_cost : float
        Weight for electricity cost objective.
    gamma_carbon : float
        Weight for carbon emission objective.
    """

    def __init__(self, batt_cap = BATTERY_CAPACITY, max_power=CHARGING_RATE, 
                 min_soc=MIN_SOC, max_soc=MAX_SOC,
                 ch_eff=CHARGING_EFFICIENCY, win_len=WINDOW_LENGTH, 
                 gamma_peak=1.0, gamma_cost=1.0, gamma_carbon=1.0):
        
        self.batt_cap = batt_cap
        self.max_power = max_power
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.ch_eff = ch_eff
        self.win_len = win_len
        self.gamma_peak = gamma_peak
        self.gamma_cost = gamma_cost
        self.gamma_carbon = gamma_carbon
    
    def get_building_energy_demand(self):
        building_energy_demand_df = pd.read_csv(f'../data/processed/building_data.csv')
        building_energy_demand_df.set_index('Datetime', inplace=True)
        
        building_energy_demand_df = building_energy_demand_df.apply(pd.to_numeric, errors='coerce')
        
        return building_energy_demand_df.values.flatten()
    
    def get_electricity_price(self):
        electricity_price_df = pd.read_csv(f'../data/processed/electricitycostG2B_data.csv')
        electricity_price_df.set_index('Datetime', inplace=True)

        electricity_price_df = electricity_price_df.apply(pd.to_numeric, errors='coerce')

        return electricity_price_df.values.flatten()

    def get_photovoltaic_generation(self):
        radiation_df = pd.read_csv(f'../data/processed/radiation_data.csv')
        temperature_df = pd.read_csv(f'../data/processed/temperature_data.csv')

        radiation_df.set_index('Datetime', inplace=True)
        temperature_df.set_index('Datetime', inplace=True)
        
        radiation_df = radiation_df.apply(pd.to_numeric, errors='coerce')
        temperature_df = temperature_df.apply(pd.to_numeric, errors='coerce')
        
        efficiency = 0.2

        pv_generation = [efficiency * PV_AREA * (rad * 0.2778) * (1 - (0.005 * (temp - 25))) for rad, temp in zip(radiation_df.values.flatten(), temperature_df.values.flatten())]

        return pv_generation

    def optimize(self, index, available, t_arr, t_dep, soc_arr, soc_dep):
        
        # for i in range(len(available)):  
        #     print(f"Batt num: {i}")
        #     print(f"Available: {available[i]}")
        #     print(f"Arrival Time: {t_arr[i]}")
        #     print(f"Departure Time: {t_dep[i]}")
        #     print(f"SOC Arrival: {soc_arr[i]}")
        #     print(f"SOC Departure: {soc_dep[i]}")
            
        """
        Optimize V2B strategy.

        Parameters:
        -----------
        bldg : list
            Building energy demand per time step (kW).
        elec : list
            Electricity price per time step (NTD/kWh).
        carbon : list
            Carbon emissions per time step (kg CO2eq/kWh).
        pv : list
            PV energy generation per time step (kW).
        t_arr : list
            List of vehicle arrival times.
        t_dep : list
            List of vehicle departure times.
        soc_arr : list
            SOC of vehicles upon arrival.
        soc_dep : list
            SOC required at departure.
        available : list of lists
            Availability of vehicles per time step.

        Returns:
        --------
        solution : list
            Optimized grid demand values after V2B operation.
        """

        bldg = self.get_building_energy_demand()[index + 24: index + 24 + self.win_len]
        elec = self.get_electricity_price()[index + 24: index + 24 + self.win_len]
        pv = self.get_photovoltaic_generation()[index + 24: index + 24 + self.win_len]

        # M = self.max_power  # Maximum charging/discharging power
        n_veh = len(soc_arr)  # Number of batteries
        epsilon = 1e-6

        # Create a new model
        model = gp.Model("V2B_Optimization")

        # Define variables
        gd_t = model.addVars(self.win_len, lb=0, vtype=GRB.CONTINUOUS, name="Grid demand at t")
        loss_t = model.addVars(self.win_len, lb=0, vtype=GRB.CONTINUOUS, name="Loss at t")

        soc_t_v = model.addVars(self.win_len, n_veh, lb=self.min_soc, ub=self.max_soc, vtype=GRB.CONTINUOUS, name="SOC for vehicle")
        ch_t_v = model.addVars(self.win_len, n_veh, lb=0, vtype=GRB.CONTINUOUS, name="Vehicle charging")
        dis_t_v = model.addVars(self.win_len, n_veh, lb=0, vtype=GRB.CONTINUOUS, name="Vehicle discharging")

        alpha_ch_t_v = model.addVars(self.win_len, n_veh, vtype=GRB.BINARY, name="Charging Indicator")
        alpha_dis_t_v = model.addVars(self.win_len, n_veh, vtype=GRB.BINARY, name="Discharging Indicator")
    
        cost_obj = self.compute_cost_objective(gd_t, elec)

        # ==============================
        # OBJECTIVE FUNCTION
        # ==============================
        
        model.setObjective(sum(cost_obj[t] for t in range(self.win_len)), GRB.MINIMIZE)

        # ==============================
        # CONSTRAINTS
        # ==============================
        model.addConstrs((alpha_ch_t_v[t, v] + alpha_dis_t_v[t, v] <= available[v][t] for t in range(self.win_len) for v in range(n_veh)), "Only Charge or Discharge at Available Hour")
        
        model.addConstrs((gp.quicksum(alpha_ch_t_v[t, v] for v in range(n_veh)) * 
                          gp.quicksum(alpha_dis_t_v[t, v] for v in range(n_veh)) == 0 
                          for t in range(self.win_len)), 
                         "Mutual Exclusivity of Charging and Discharging Across Vehicles")
        
        model.addConstrs((soc_t_v[int(t_dep[v]) - 1, v] == soc_dep[v] for v in range(n_veh)), "Leaving SOC")
        model.addConstrs((soc_t_v[0, v] == soc_arr[v] + (ch_t_v[0, v] / self.batt_cap) * self.ch_eff - (dis_t_v[0, v] / self.batt_cap) / self.ch_eff for v in range(n_veh)), "Arriving SOC")
        
        model.addConstrs((soc_t_v[t, v] == soc_t_v[t - 1, v] + (ch_t_v[t, v] / self.batt_cap) * self.ch_eff - (dis_t_v[t, v] / self.batt_cap) / self.ch_eff for t in range(1, self.win_len) for v in range(n_veh)), "Update SOC")
    
        model.addConstrs((sum(ch_t_v[t, v] for v in range(n_veh)) <= self.max_power for t in range(self.win_len)), "Max Charging Power")
        model.addConstrs((sum(dis_t_v[t, v] for v in range(n_veh)) <= self.max_power for t in range(self.win_len)), "Max Discharging Power")
        
        # Grid demand equation
        model.addConstrs((gd_t[t] == bldg[t] - pv[t] + sum(ch_t_v[t, v] - dis_t_v[t, v] for v in range(n_veh)) + loss_t[t] for t in range(self.win_len)), "Update Grid Demand")
        
        # ==============================
        # RUN OPTIMIZATION
        # ==============================


        print("="*10, f"Index {index}", "="*10)
        
        # Set number of threads used
        # model.Params.Threads = 1  # Set to desired number of threads
        # print(f"Number of threads used: {model.Params.Threads}")
        
        model.optimize()

        if model.status == GRB.OPTIMAL:
            initial_demand = [max(bldg[t] - pv[t], 0) for t in range(self.win_len)]
            final_grid_demand = [gd_t[t].x for t in range(self.win_len)]  
            charging_demand = [sum([ch_t_v[t, v].x - dis_t_v[t, v].x for v in range(n_veh)]) for t in range(self.win_len)]
            
            optimal_objective = model.objVal
            optimization_time = model.Runtime
            mip_gap = model.MIPGap * 100 if model.isMIP else '-'  # Only relevant for MIP problems
            
            # --- Clean up after each run ---
            gp.disposeDefaultEnv()  # releases Gurobi environment from memory
            gc.collect()            # trigger Python garbage collector
            time.sleep(0.8)           # wait 0.8 seconds before next optimization
            
            return initial_demand, final_grid_demand, charging_demand, pv
        
        elif model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write(f"infeasible_model.ilp")
            return None
            
        else:
            print(f"Optimization ended with status {model.status}")
            return None
        
    def compute_cost_objective(self, gd_t, electricity_cost):
        """Compute electricity cost objective."""
        cost_ob = {t: electricity_cost[t] * gd_t[t] for t in range(self.win_len)}
        total_cost = gp.quicksum(cost_ob.values())  # Total electricity cost
        return cost_ob
