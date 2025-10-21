from math import e
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
from util.batteryreader import get_battery_details
from tqdm import tqdm

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
                 proj_name=PROJECT_NAME):
        
        self.batt_cap = batt_cap
        self.max_power = max_power
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.ch_eff = ch_eff
        self.win_len = win_len
        self.proj_name = proj_name
        self.output_dir = os.path.join(".", "output", self.proj_name)
        
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

    def run_single_optimization(self, index, available, t_arr, t_dep, soc_arr, soc_dep):

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

        print("-"*60)
        print("Starting optimization...")
        
        # Set number of threads used
        # model.Params.Threads = 1  # Set to desired number of threads
        # print(f"Number of threads used: {model.Params.Threads}")
        
        model.optimize()

        if model.status == GRB.OPTIMAL:
            initial_demand = [max(bldg[t] - pv[t], 0) for t in range(self.win_len)]
            final_grid_demand = [gd_t[t].x for t in range(self.win_len)]  
            charging_demand = [sum([ch_t_v[t, v].x - dis_t_v[t, v].x for v in range(n_veh)]) for t in range(self.win_len)]
            soc_t_v = [[soc_t_v[t, v].x for t in range(self.win_len)] for v in range(n_veh)]

            total_grid_demand = sum(final_grid_demand)
            total_demand = sum([bldg[t] for t in range(self.win_len)]) + sum([ch_t_v[t, v].x for v in range(n_veh) for t in range(self.win_len)])

            pv_loss = sum([loss_t[t].x for t in range(self.win_len)])
            pv_generated = sum([pv[t] for t in range(self.win_len)])
            
            self_sufficiency = 1 - (total_grid_demand / total_demand) if total_demand > 0 else 0
            pv_utilization = (pv_generated - pv_loss) / pv_generated if pv_generated > 0 else 0

            print("Optimization completed successfully.")
            
            optimal_objective = model.objVal
            optimization_time = model.Runtime
            mip_gap = model.MIPGap * 100 if model.isMIP else '-'  # Only relevant for MIP problems

            return np.array(initial_demand), np.array(final_grid_demand), np.array(charging_demand), np.array(soc_t_v), optimal_objective, optimization_time, mip_gap, self_sufficiency, pv_utilization

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

    def run_full_optimization(self):

        index, available, SOC_a_v, SOC_d_v, t_a_v, t_d_v = get_battery_details(window_length = self.win_len)

        sys.stdout = open(f'./outputlog/output_log_{self.proj_name}.txt', 'w')

        os.makedirs(self.output_dir, exist_ok=True)
        
        initial_demands = []
        final_grid_demands = []
        charging_demands = []
        soc_t_v_s = []
        optimal_objectives = []
        optimization_times = []
        mip_gaps = []
        self_sufficiencies = []
        pv_utilizations = []

        for idx in tqdm(range(len(available)), desc='Processing series', unit='series'):  
            
            print("="*60)
            print(f"Starting series num: {idx}, Datetime Index: {index[idx]}")
            
            print("-"*60)
            print("Battery Details:")
            print("-"*60)

            available_current = np.array(available[idx])
            SOC_a_v_current = np.array(SOC_a_v[idx])
            SOC_d_v_current = np.array(SOC_d_v[idx])
            t_a_v_current = np.array(t_a_v[idx])
            t_d_v_current = np.array(t_d_v[idx])
            
            print(f"Number of available batteries:\n{len(available_current)}")
            print("-"*20)
            print(f"Time arrival:\n{t_a_v_current}")
            print("-"*20)
            print(f"Time departure:\n{t_d_v_current}")
            print("-"*20)
            print(f"State of Charge (SoC) at the beginning of availability:\n{SOC_a_v_current}")
            print("-"*20)
            print(f"State of Charge (SoC) at the end of availability:\n{SOC_d_v_current}")
            

            initial_demand, final_grid_demand, charging_demand, soc_t_v, optimal_objective, optimization_time, mip_gap, self_sufficiency, pv_utilization = self.run_single_optimization(
                index[idx], available[idx], t_a_v[idx], t_d_v[idx], SOC_a_v[idx], SOC_d_v[idx]
            )

            initial_demands.append(initial_demand)
            final_grid_demands.append(final_grid_demand)
            charging_demands.append(charging_demand)
            soc_t_v_s.append(soc_t_v)
            optimal_objectives.append(optimal_objective)
            optimization_times.append(optimization_time)
            mip_gaps.append(mip_gap)
            self_sufficiencies.append(self_sufficiency)
            pv_utilizations.append(pv_utilization)

            print("-"*60)
            print("Optimization Results")

            print(f"Initial Demand:\n{np.round(initial_demand, 2)}")
            print("-"*20)
            print(f"Final Grid Demand:\n{np.round(final_grid_demand, 2)}")
            print("-"*20)
            print(f"Charging Demand:\n{np.round(charging_demand, 2)}")
            print("-"*20)
            print(f"Battery State of Charge (SoC):\n{np.round(soc_t_v, 2)}")
            print("-"*20)
            print(f"Optimal Objective Value: {optimal_objective}")
            print(f"Optimization Time (s): {optimization_time}")
            print(f"MIP Gap (%): {mip_gap}")
            print(f"Self-Sufficiency: {self_sufficiency:.4f}")
            print(f"PV Utilization: {pv_utilization:.4f}")
            
            print("="*20, f"Series num: {idx} ENDED", "="*20, '\n\n')
        
        sys.stdout.close()
        sys.stdout = sys.__stdout__

        max_batt = max([soc_t_v.shape[0] for soc_t_v in soc_t_v_s])
        
        soc_t_v_s = [np.pad(arr, ((0, max_batt - arr.shape[0]),(0, 0)), mode='constant', constant_values=np.nan) for arr in soc_t_v_s]
        
        # Save results
        np.save(os.path.join(self.output_dir, f'initial_demands.npy'), np.array(initial_demands))
        np.save(os.path.join(self.output_dir, f'final_grid_demands.npy'), np.array(final_grid_demands))
        np.save(os.path.join(self.output_dir, f'charging_demands.npy'), np.array(charging_demands))
        np.save(os.path.join(self.output_dir, f'soc_t_v.npy'), np.array(soc_t_v_s))
        np.save(os.path.join(self.output_dir, f'optimal_objectives.npy'), np.array(optimal_objectives))
        np.save(os.path.join(self.output_dir, f'optimization_times.npy'), np.array(optimization_times))
        np.save(os.path.join(self.output_dir, f'mip_gaps.npy'), np.array(mip_gaps))
        np.save(os.path.join(self.output_dir, f'self_sufficiencies.npy'), np.array(self_sufficiencies))
        np.save(os.path.join(self.output_dir, f'pv_utilizations.npy'), np.array(pv_utilizations))

        return initial_demands, final_grid_demands, charging_demands, soc_t_v_s, optimal_objectives, optimization_times, mip_gaps, self_sufficiencies, pv_utilizations