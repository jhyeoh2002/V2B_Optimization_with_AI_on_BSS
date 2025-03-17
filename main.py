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
from config import PROJECT_NAME, WINDOW_LENGTH, ITERATIONS, BATTERY_CAPACITY, BATTERY_COST, BATTERY_CYCLE_LIFE, DEPTH_OF_DISCHARGE, CHARGING_RATE, CHARGING_EFFICIENCY, DEPARTURE_SOC, MIN_SOC, MAX_SOC, BATTERIES_PER_STATION, NUM_STATIONS, TOTAL_BATTERIES, DAYS, TIME_STEPS
print(PROJECT_NAME)

# Now you can use these variables in your code
def main():

    data_path = './Data/Full_Data.csv'
    batteryinfo_path = './Data/Battery_info.csv'

    data = pd.read_csv(data_path)
    batteryinfo = pd.read_csv(batteryinfo_path)

    BldgEnCon = data['energy (kWh)']
    Temp = data['Temperature,T (deg)']
    Rad = data['Radiation Intensity, I']
    CarbInt = data['Carbon Intensity (kgC02eq/kWh)']

    # Choose the start day
    start_date = '2024-02-01' 
    end_date = '2024-02-07' 

    chosen_dateli = []

    date_range = pd.date_range(start=start_date, end=end_date)
    for chosen_date in date_range:
        chosen_dateli.append(chosen_date.strftime('%Y-%m-%d'))

    a_vt = batteryinfo['Availability']
    t_a_v = batteryinfo['Arrival_hour']
    t_d_v = batteryinfo['Departure_hour']
    SOC_a_v = batteryinfo['Arrival_SOC']
    SOC_d_v = batteryinfo['Departure_SOC']    
    D_B_t = data['energy (kWh)']

    projPath = './results/' + PROJECT_NAME

    if not os.path.exists(projPath):
        os.makedirs(projPath + '/figures')

    BldgEnCon = pd.read_csv(r"Data/Cleaned data/Building Energy Consumption.csv").set_index('Datetime')
    BldgEnCon = np.array(BldgEnCon.iloc[:,4])

    PVGen = pd.read_csv(r"Data/Cleaned data/PV generation.csv").set_index('Datetime')
    PVGen = np.array(PVGen.iloc[:,4])

    CarbInt = pd.read_csv(r"Data/Cleaned data/Carbon Intensity Data 2020.csv").set_index('Datetime')
    CarbInt = np.array(CarbInt.iloc[:,0])

    return BldgEnCon, PVGen, CarbInt

# if __name__ == "__main__":
#     main()
