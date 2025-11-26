import gurobipy as gp
from gurobipy import GRB
from logstatus import log_status

import sys
import os
import numpy as np
from optimization.util.batteryreader import get_battery_details

sys.path.append(os.path.abspath(".."))

import config as cfg

# ==============================
# VEHICLE-TO-BUILDING OPTIMIZATION FUNCTION
# ==============================

class GurobiOptimizer:

    def __init__(self):
        """
        Initialize the optimizer with configuration parameters.
        """
        self.batt_cap = cfg.BATTERY_CAPACITY  # Battery capacity in kWh
        self.max_power = cfg.CHARGING_RATE  # Maximum charging/discharging power in kW
        self.min_soc = cfg.MIN_SOC  # Minimum state of charge
        self.max_soc = cfg.MAX_SOC  # Maximum state of charge
        self.ch_eff = cfg.CHARGING_EFFICIENCY  # Charging efficiency
        self.win_len = cfg.WINDOW_LENGTH  # Optimization window length

        # Flag to indicate whether Gurobi is available in this runtime
        self._gurobi_available = gp is not None

    def _require_gurobi(self):
        """
        Ensure that Gurobi is available before proceeding.
        """
        if not self._gurobi_available:
            raise RuntimeError("Gurobi is not available in this runtime.")

    def optimize(self, availability, t_dep, soc_arr, soc_dep, bldg, elec_G2B, elec_G2V, pv):
        """
        Run a single optimization for vehicle-to-building energy allocation.

        Parameters:
            availability (list): Vehicle availability matrix.
            t_dep (list): Departure times for vehicles.
            soc_arr (list): Initial state of charge for vehicles.
            soc_dep (list): Desired state of charge at departure.
            bldg (list): Building energy demand.
            elec_G2B (list): Electricity price for grid-to-building.
            elec_G2V (list): Electricity price for grid-to-vehicle.
            pv (list): Photovoltaic generation data.

        Returns:
            dict: Optimization results including time-series data and summary metrics.
        """
        # Ensure Gurobi is available before attempting to build a model
        self._require_gurobi()

        # Load Parameters and Data
        M = self.max_power  # Maximum charging/discharging power
        n_veh = len(soc_arr)  # Number of vehicles
        epsilon = 1e-6  # Small constant to avoid numerical issues

        # Create a new Gurobi model
        model = gp.Model("V2B_Optimization")

        # Set model parameters
        model.setParam("Threads", 8)  # Use 8 threads for parallel optimization
        model.setParam("MIPGap", 0.01)  # Set a 1% MIP gap

        # ==============================
        # DEFINE VARIABLES
        # ==============================

        # Grid demand and loss variables
        gd_t = model.addVars(self.win_len, lb=0, vtype=GRB.CONTINUOUS, name="Grid demand at t")
        loss_t = model.addVars(self.win_len, lb=0, vtype=GRB.CONTINUOUS, name="Loss at t")

        # Energy flow variables
        G2B_t = model.addVars(self.win_len, lb=0, vtype=GRB.CONTINUOUS, name="Grid to Building")
        P2B_t = model.addVars(self.win_len, lb=0, vtype=GRB.CONTINUOUS, name="PV to Building")

        G2V_t_v = model.addVars(self.win_len, n_veh, lb=0, vtype=GRB.CONTINUOUS, name="Grid to Vehicle")
        P2V_t_v = model.addVars(self.win_len, n_veh, lb=0, vtype=GRB.CONTINUOUS, name="PV to Vehicle")

        V2B_t_v = model.addVars(self.win_len, n_veh, lb=0, vtype=GRB.CONTINUOUS, name="Vehicle to Building")

        # State of charge (SOC) variables
        soc_t_v = model.addVars(self.win_len, n_veh, lb=self.min_soc, ub=self.max_soc, vtype=GRB.CONTINUOUS, name="SOC for vehicle")

        # Binary indicators for charging and discharging
        alpha_ch_t = model.addVars(self.win_len, vtype=GRB.BINARY, name="Charging Indicator")
        alpha_dis_t = model.addVars(self.win_len, vtype=GRB.BINARY, name="Discharging Indicator")

        # Unmet demand and cost variables
        unmet_v = model.addVars(n_veh, lb=0.0, vtype=GRB.CONTINUOUS, name="Shortfall_v")
        cost_obj = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Cost Objective")

        # ==============================
        # OBJECTIVE FUNCTION
        # ==============================

        # Minimize the weighted sum of unmet demand and cost
        model.setObjective(10e5 * sum(unmet_v[v] for v in range(n_veh)) + cost_obj, GRB.MINIMIZE)

        # ==============================
        # CONSTRAINTS
        # ==============================

        # Energy balance constraints
        model.addConstrs((G2B_t[t] + sum(G2V_t_v[t, v] for v in range(n_veh)) == gd_t[t] for t in range(self.win_len)), "Grid Energy Balance")
        model.addConstrs((P2B_t[t] + sum(P2V_t_v[t, v] for v in range(n_veh)) == pv[t] - loss_t[t] for t in range(self.win_len)), "PV Energy Balance")
        model.addConstrs((G2B_t[t] + P2B_t[t] + sum(V2B_t_v[t, v] for v in range(n_veh)) == bldg[t] for t in range(self.win_len)), "Building Energy Balance")

        # Charging/discharging constraints
        model.addConstrs((G2V_t_v[t, v] + P2V_t_v[t, v] + V2B_t_v[t, v] <= M * availability[v][t] for t in range(self.win_len) for v in range(n_veh)), "Only Charge or Discharge at Available Hour")
        model.addConstrs((alpha_ch_t[t] + alpha_dis_t[t] <= 1 for t in range(self.win_len)), "Only Charge or Discharge at Time t")

        # Link charging and discharging binaries with their continuous powers
        model.addConstrs((sum(G2V_t_v[t, v] + P2V_t_v[t, v] for v in range(n_veh)) <= M * alpha_ch_t[t] for t in range(self.win_len)), "LinkChargeBinary")
        model.addConstrs((sum(V2B_t_v[t, v] for v in range(n_veh)) <= M * alpha_dis_t[t] for t in range(self.win_len)), "LinkDischargeBinary")

        # SOC updating constraints
        model.addConstrs((soc_t_v[int(t_dep[v]) - 1, v] + (unmet_v[v]/self.batt_cap) >= soc_dep[v] for v in range(n_veh)), "Leaving SOC")
        model.addConstrs((soc_t_v[0, v] == soc_arr[v] + (G2V_t_v[0, v] + P2V_t_v[0, v]) / self.batt_cap * self.ch_eff - (V2B_t_v[0, v] / self.batt_cap) / self.ch_eff for v in range(n_veh)), "Arriving SOC")
        model.addConstrs((soc_t_v[t, v] == soc_t_v[t - 1, v] + (G2V_t_v[t, v] + P2V_t_v[t, v]) / self.batt_cap * self.ch_eff - (V2B_t_v[t, v] / self.batt_cap) / self.ch_eff for t in range(1, self.win_len) for v in range(n_veh)), "Update SOC")

        # Power limits
        model.addConstrs((sum(G2V_t_v[t, v] + P2V_t_v[t, v] for v in range(n_veh)) <= self.max_power for t in range(self.win_len)), "Max Charging Power")
        model.addConstrs((sum(V2B_t_v[t, v] for v in range(n_veh)) <= self.max_power for t in range(self.win_len)), "Max Discharging Power")

        # Cost objective definition
        model.addConstr(cost_obj == sum(elec_G2V[t] * sum(G2V_t_v[t, v] for v in range(n_veh)) + elec_G2B[t] * G2B_t[t] for t in range(self.win_len)), "Cost Objective Definition")

        # ==============================
        # RUN OPTIMIZATION
        # ==============================
        print("-" * 60)
        print("Starting optimization...")
        log_status("Before solve")
        model.optimize()
        log_status("After solve")

        # ==============================
        # POST-PROCESS RESULTS
        # ==============================
        if model.status == GRB.OPTIMAL:
            print("Optimization completed successfully.")
            print(f"Optimal Objective Value: {model.ObjVal:.4f}")
            print(f"Solve Time (s): {model.Runtime:.2f}")
            if model.isMIP:
                print(f"MIP Gap: {model.MIPGap * 100:.3f}%")

            # Extract variable values
            gd_t_val = np.array([gd_t[t].x for t in range(self.win_len)])
            loss_t_val = np.array([loss_t[t].x for t in range(self.win_len)])
            G2B_val = np.array([G2B_t[t].x for t in range(self.win_len)])
            P2B_val = np.array([P2B_t[t].x for t in range(self.win_len)])

            G2V_val = np.array([[G2V_t_v[t, v].x for v in range(n_veh)] for t in range(self.win_len)])
            P2V_val = np.array([[P2V_t_v[t, v].x for v in range(n_veh)] for t in range(self.win_len)])
            V2B_val = np.array([[V2B_t_v[t, v].x for v in range(n_veh)] for t in range(self.win_len)])
            SOC_val = np.array([[soc_t_v[t, v].x for v in range(n_veh)] for t in range(self.win_len)])

            unmet_val = np.array([unmet_v[v].x for v in range(n_veh)])
            total_cost = cost_obj.x

            # Derived performance metrics
            initial_demand = np.maximum(bldg - pv, 0)
            charging_demand = np.sum(G2V_val + P2V_val, axis=1)

            total_grid_demand = np.sum(gd_t_val)
            total_demand = np.sum(G2B_val + P2B_val) + np.sum(G2V_val + P2V_val)
            pv_generated = np.sum(pv)
            pv_loss = np.sum(loss_t_val)

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
                "grid_demand": gd_t_val, # Grid demand over time (window length)
                "pv_loss": loss_t_val, # PV loss over time (window length)
                "initial_demand": initial_demand, # Initial demand (window length)
                "charging_demand": charging_demand, # Charging demand (window length)
                
                "G2B": G2B_val, # Grid to Building flows (window length)
                "P2B": P2B_val, # PV to Building flows (window length)
                
                "G2V": G2V_val, # Grid to Vehicle flows (n_vehicle x window length)
                "P2V": P2V_val, # PV to Vehicle flows (n_vehicle x window length)
                "V2B": V2B_val, # Vehicle to Building flows (n_vehicle x window length)

                "SOC": SOC_val, # State of Charge over time (n_vehicle x window length)
                "unmet": unmet_val, # Unmet demand per vehicle (n_vehicle)

                "objective_value": model.ObjVal, # Final objective value (scalar)
                "optimization_time_s": model.Runtime, # Optimization time in seconds (scalar)
                "mip_gap_percent": model.MIPGap * 100 if model.isMIP else None, # MIP gap percentage (scalar)
                
                "total_cost": total_cost, # Total cost incurred (scalar)
                "self_sufficiency": self_sufficiency, # Self-sufficiency metric (scalar)
                "pv_utilization": pv_utilization, # PV utilization metric (scalar)
            }

            return result

        else:
            
            print(f"Optimization did not reach optimality. Status: {model.status}")
            
            return None

