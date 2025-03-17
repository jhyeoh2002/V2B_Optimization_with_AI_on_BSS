# ====== Optimization Parameters ======

# Project settings
PROJECT_NAME = 'test'
WINDOW_LENGTH = 6  # Hours
ITERATIONS = 1

# Battery parameters
BATTERY_CAPACITY = 1.5  # kWh
BATTERY_COST = 9071  # $/kWh
BATTERY_CYCLE_LIFE = 2020  # cycles
DEPTH_OF_DISCHARGE = 0.7

# Charging parameters
CHARGING_RATE = 14  # kW
CHARGING_EFFICIENCY = 0.9
DEPARTURE_SOC = 0.7
MIN_SOC = 0.2
MAX_SOC = 0.9

# Station parameters
BATTERIES_PER_STATION = 38
NUM_STATIONS = 2
TOTAL_BATTERIES = BATTERIES_PER_STATION * NUM_STATIONS

# Time parameters
DAYS = 45
TIME_STEPS = DAYS * 24  # 24 hours per day