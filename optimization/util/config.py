# ====== Optimization Parameters ======

# Project Settings
WINDOW_LENGTH = 48  # hours
PV_AREA = 700  # m^2

PROJECT_NAME = f'V3_With_G2Vcost_WL{WINDOW_LENGTH}_PV{PV_AREA}'

# Battery Parameters
BATTERY_CAPACITY = 1.5  # kWh
BATTERY_COST = 9071     # $/kWh
BATTERY_CYCLE_LIFE = 2020  # cycles

# Charging Parameters
CHARGING_RATE = 14      # kW
CHARGING_EFFICIENCY = 0.9
DEPARTURE_SOC = 0.9
MIN_SOC = 0.2
MAX_SOC = 0.9

# Station Parameters
BATTERIES_PER_STATION = 38
NUM_STATIONS = 2
TOTAL_BATTERIES = BATTERIES_PER_STATION * NUM_STATIONS
