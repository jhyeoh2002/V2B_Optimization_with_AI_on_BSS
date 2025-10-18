import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import pandas as pd
import pandas as pd 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from util.GurobiOptimizerV2B import GurobiOptimizerV2B
from optimization.util.config import PROJECT_NAME, WINDOW_LENGTH, ITERATIONS, BATTERY_CAPACITY, BATTERY_COST, BATTERY_CYCLE_LIFE, DEPTH_OF_DISCHARGE, CHARGING_RATE, CHARGING_EFFICIENCY, DEPARTURE_SOC, MIN_SOC, MAX_SOC, BATTERIES_PER_STATION, NUM_STATIONS, TOTAL_BATTERIES, DAYS, TIME_STEPS, DATA_PATH, BATTERYINFO_PATH

# Now you can use these variables in your code
def main():
    projPath = './results/' + PROJECT_NAME

    if not os.path.exists(projPath):
        os.makedirs(projPath + '/figures')
        
    data = pd.read_csv(DATA_PATH)
    batteryinfo = pd.read_csv(BATTERYINFO_PATH)

    # Import necessary columns
    BldgEnCon = data['energy (kWh)']
    Temp = data['Temperature,T (deg)']
    Rad = data['Radiation Intensity, I']
    CarbInt = data['Carbon Intensity (kgC02eq/kWh)']
    PVGen = data['PV Generation (kWh)']
    BldgCost = data['Building Electricity Cost (NTD/kWh)']
    VehCost = data['Vehicle Electricity Cost (NTD/kWh)'] 
    Available = data['Available'] #available battery at the time in station
    
    t_a_v = batteryinfo['Arrival_hour']
    SOC_a_v = batteryinfo['Arrival_SOC']



if __name__ == "__main__":
    main()
