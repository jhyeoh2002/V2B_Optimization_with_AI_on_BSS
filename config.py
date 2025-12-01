from encodings.punycode import T
import numpy as np
import os

# ==============================================================================
# 1. Project & Time Configuration
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

BATTERYDEMAND_PATH = "battery_demandV2"

# ==============================================================================
# 2. File Paths
# ==============================================================================
# Formatted strings for data loading
A_PATH_TEMPLATE = './data/raw/Gogoro/台北市大安區_臺大二活停車場站A ({:02d}).csv'
B_PATH_TEMPLATE = './data/raw/Gogoro/台北市大安區_臺大二活停車場站B ({:02d}).csv'

# ==============================================================================
# 3. System Specifications (Physical)
# ==============================================================================
# --- Battery Specs ---
BATTERY_CAPACITY = 1.5      # kWh
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
WINDOW_SIZE = 18 # Used for general windowing

# Seed 48 Configuration
WINDOW_SIZE_48 = 48
SEED_48 = np.array([
    28.0, 26.0, 22.0, np.nan, np.nan, 21.0, 28.0, 17.0, 32.0, 36.0, 30.0, 19.0, 
    20.0, np.nan, np.nan, 33.0, 37.0, 42.0, 42.0, 42.0, 38.67, 33.33, 30.8, 
    27.33, 29.2, 30.0, 29.6, 26.0, np.nan, 26.67, 16.0, 21.0, 20.67, 25.5, 
    26.0, 22.0, 16.33, 22.0, 32.0, 35.0, 40.0, 40.0, 40.0, 38.0, np.nan, 
    6.0, 13.0, 19.33
])

# ==============================================================================
# 5. Optimization & Sampling Parameters
# ==============================================================================
WINDOW_LENGTH = WINDOW_SIZE  # hours
BANDWIDTH = 0.1
TOP_K = 1000

# --- Sampling ---
BOOTSTRAP_SIZE = 1
NSAMPLE = 200
N_EXTENDED = 2000

# --- Tolerances ---
TOLERANCE_DEFAULT = 15      # General tolerance (up to 1/3 of window)

# ==============================================================================
# 6. Output Naming Configuration
# ==============================================================================
PROJECT_DETAIL = f"Test"
PROJECT_NAME = f'WL{WINDOW_LENGTH}_PV{PV_AREA}_{PROJECT_DETAIL}'

# ==============================================================================