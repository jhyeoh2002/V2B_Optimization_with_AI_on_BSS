from encodings.punycode import T
import numpy as np
import os

from pyparsing import C

# ==============================================================================
# 1. Project Configuration
# ==============================================================================
START_DATE = "2023-01-01 00:00:00"
END_DATE = "2024-09-30 23:00:00"
RESOLUTION = '1h'

TEST_MODE = False
TEST_DATE1 = ['2024-03-15', '2024-03-16']
TEST_DATE2 = ['2024-03-29', '2024-03-30']
CASE0 = "case0_test"
CASE1 = "case1_real_only"
CASE2 = "case2_nan_filled"
CASE3 = "case3_extended_generated"

BATTERYDEMAND_DIR = "data/battery_demandV3"
BATTERY_FILE = f"{BATTERYDEMAND_DIR}/resample_full.csv"
TS_DIR = "data/timeseries"

OPTRESULTS_DIR = "data/optimization_resultsV3"

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
PV_AREA = 1200  # m^2

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

import torch
import os

# ==========================================
# SYSTEM & HARDWARE
# ==========================================
SEED = 42
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DATA PROCESSING
# ==========================================
SEQUENCE_LENGTH = 24     # The lookback period for the model
TOLERANCE = 4            # Tolerance for data merging logic

# Paths
BASE_OPT_FOLDER = "./data/optimization_results"
TRAIN_RESULTS_DIR = "./data/training_results"
TRAIN_FIGURE_DIR = "./figures/training_results"

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
RUN_NAME = f"STAF_currentSOC_V8"

EPOCHS = 5000
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
# WEIGHT_DECAY = LEARNING_RATE

# Learning Rate Scheduler
GAMMA = 0.5             # Factor to reduce LR by
LR_PATIENCE = 50         # Epochs to wait before reducing LR

# Early Stopping
ES_PATIENCE = 100        # Epochs to wait before stopping completely

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
EMBEDDING_DIM = 8
NUM_EMBEDDINGS = 1000  
N_HEADS = 4
# Recommended (Smoother funnel)
HIDDEN_DIM_1 = 128  # Reduced from 1024
HIDDEN_DIM_2 = 32  # Reduced from 256
DROPOUT = 0.6
ATTENTION_DROPOUT = 0.6