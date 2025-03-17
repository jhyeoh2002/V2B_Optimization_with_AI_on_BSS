import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt

from functions.optimization import Optimization
from functions.readcsv import readData

from scipy.stats import truncnorm

# TODO Set parameters (Modifiable)

#Set a name for this run
projName = 'tempfull'

#Charging Rate and Efficiency
charging_rate = 14       #kW   #Charging rate for gogoro station
charging_eff = 0.9

#Boundaries for Batteries
EVLowerBoundSoC = 0.2
EVUpperBoundSoC = 0.9

#Set number of iteration for basinhopping
iteration = 1

c_Batt = 9071 # Battery cost per kilowatt-hour ($/kWh)
pi_Cap = 1.5 # Battery energy capacity (kWh)
pi_CL = 2020 # Battery lifetime in terms of cycle life
pi_DoD = 0.7 # DoD for a certain cycle life
n_station = 38*2 # Number of batteries at the station
SOC_thr = 0.7 # Required leaving SOC

# Choose tnum
days = int(45)
tnum = int(days*24) # 24 for a day optimization

data = pd.read_csv('./Data/Full_Data.csv')
batteryinfo = pd.read_csv('./Data/Battery_info.csv')

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

# global D_B_t, S_R_t, N_V_t, n_total, t_a_v, t_d_v, a_vt, SOC_a_v, SOC_d_v, pi_Cn_t, c_G2B_t, c_G2V_t

a_vt = batteryinfo['Availability']
t_a_v = batteryinfo['Arrival_hour']
t_d_v = batteryinfo['Departure_hour']
SOC_a_v = batteryinfo['Arrival_SOC']
SOC_d_v = batteryinfo['Departure_SOC']    
D_B_t = data['energy (kWh)']

date = pd.date_range(start='2023-06-16', end='2023-06-22', freq='D')
date_strings = date.strftime('%Y-%m-%d').tolist()

projPath = './results/' + projName

if not os.path.exists(projPath):
    os.makedirs(projPath + '/npyFiles')
    os.makedirs(projPath + '/csv')
    os.makedirs(projPath + '/figures')

####################### V2B Charging/Discharging Slots #######################

V2B = Optimization(chargingRate = charging_rate, chargingEff = charging_eff , 
                   gammaPeak = 1/3, gammaCost = 1/3, gammaCarbon = 1/3, 
                   lowestSoC = EVLowerBoundSoC, highestSoC = EVUpperBoundSoC, projName=projName)

window_length = 6
import pandas as pd 
import numpy as np

def readData() -> tuple[np.array, np.array, np.array]:
    '''
    Read data from .csv files and return year data 2020 in array format
    
    Parameters  
    None
    
    Returns
    year data(tuple): Building Energy Consumption, PV Generation, Carbon Intensity 
    
    '''

BldgEnCon = pd.read_csv(r"Data/Cleaned data/Building Energy Consumption.csv").set_index('Datetime')
BldgEnCon = np.array(BldgEnCon.iloc[:,4])

PVGen = pd.read_csv(r"Data/Cleaned data/PV generation.csv").set_index('Datetime')
PVGen = np.array(PVGen.iloc[:,4])

CarbInt = pd.read_csv(r"Data/Cleaned data/Carbon Intensity Data 2020.csv").set_index('Datetime')
CarbInt = np.array(CarbInt.iloc[:,0])

return BldgEnCon, PVGen, CarbInt

EVChargingV2B = V2B.optimize_ev(days = days, length = window_length, iteration = iteration)
np.save(projPath + '/npyFiles/EVchargingV2B_'+str(window_length)+'.npy', EVChargingV2B)
print('Saving V2B Charging_'+str(window_length)+'/Discharging Results to ./npyFiles\n')