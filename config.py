from encodings.punycode import T
import numpy as np
import os

# ==============================================================================
# 1. Project Configuration
# ==============================================================================
START_DATE = "2023-01-01 00:00:00"
END_DATE = "2024-09-30 23:00:00"
RESOLUTION = '1h'

TEST_MODE = False
TEST_DATE1 = ['2024-03-15', '2024-03-16']
TEST_DATE2 = ['2024-03-29', '2024-03-30']

CASE1 = "case1_real_only"
CASE2 = "case2_nan_filled"
CASE3 = "case3_extended_generated"

BATTERYDEMAND_DIR = "data/battery_demand"

# ==============================================================================
# 2. DATA GENERATION PARAMETERS
# ==============================================================================
# --- Battery Specs ---
BATTERY_CAPACITY = 1.7      # kWh
BATTERY_COST = 9071         # $/kWh
BATTERY_CYCLE_LIFE = 2020   # cycles

# --- Charging Parameters ---
CHARGING_RATE = 14          # kW
CHARGING_EFFICIENCY = 0.9

# --- SOC Constraints ---
SOC_THR = 0.9       # State of Charge threshold for leaving station
DEPARTURE_SOC = 0.9
MIN_SOC = 0.2
MAX_SOC = 0.9

# --- Station Infrastructure ---
BATTERIES_PER_STATION = 38
NUM_STATIONS = 2
TOTAL_BATTERIES = BATTERIES_PER_STATION * NUM_STATIONS

# --- Renewable Energy ---
PV_AREA = 500  # m^2

# ==============================================================================
# 4. Statistical & Randomization Parameters
# ==============================================================================
MEAN = 27.91
STD = 10.34
RANDOM_STATE = 42

# --- Active Seed Configuration ---
SEED = None
WINDOW_SIZE = 36 # Used for general windowing
TOLERANCE_DEFAULT = 33      # General tolerance (up to 1/3 of window)

# ==============================================================================
# 5. Optimization & Sampling Parameters
# ==============================================================================
WINDOW_LENGTH = WINDOW_SIZE  # hours
BANDWIDTH = 0.1
TOP_K = 1000

# --- Sampling ---
BOOTSTRAP_SIZE = 50
NSAMPLE = 500
N_EXTENDED = 2000
MAX_SHIFT = 1

# --- Tolerances ---

# ==============================================================================
# 6. Output Naming Configuration
# ==============================================================================
PROJECT_DETAIL = f"Test"
PROJECT_NAME = f'WL{WINDOW_LENGTH}_PV{PV_AREA}_{PROJECT_DETAIL}'

# ==============================================================================