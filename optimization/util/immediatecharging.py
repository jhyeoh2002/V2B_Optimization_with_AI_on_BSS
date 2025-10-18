import math
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

ParkPatt = pd.read_csv(r"./Data/Cleaned data/EV Parking Pattern.csv")

def slotcal(row,rate):

    required_energy = (row['SOC_desired'] - row['SOC_initial']) * row['Battery Capacity (kWh)']
    no_of_slots = math.ceil(required_energy/rate)
    
    return required_energy, no_of_slots

def allocateslot(row,rate):

    charging_energy = [0]*24
    required_energy = row['Required Energy(kWh)']

    time = int(row['Time_in'])
    
    while required_energy > rate:
        
        charging_energy[time]+=rate
        
        time+=1
        required_energy-=rate

    charging_energy[time]+=required_energy
    
    return row['Date'], charging_energy

def get_EVimmediateCharge(charging_rate):
    
    # Perform Calculation
    ParkPatt[['Required Energy(kWh)', 'No. of Slots']] = ParkPatt.apply(
        lambda row: pd.Series(slotcal(row, charging_rate)), axis=1
    )

    ParkPatt['Hours_in_lot']=+ParkPatt['Time_out']-ParkPatt['Time_in']
    ParkPatt['energy_per_slot']=ParkPatt['Required Energy(kWh)']/ParkPatt['Hours_in_lot']

    EVChargingImmediate = np.zeros(24*366) #Create empty array for immediate charging

    results = ParkPatt.apply(lambda row: pd.Series(allocateslot(row, charging_rate)), axis=1)  #Calculate the charging time for each vehicles
    datelist = [i.strftime('%Y-%m-%d') for i in pd.date_range(datetime(2020,1,1),periods=366).tolist()] 

    for index, row in results.iterrows():
        day = datelist.index(row.iloc[0])
        EVChargingImmediate[day*24:day*24+24]+=row.iloc[1]
    
    print('Calculated Immediate Charging Slots')
    
    return EVChargingImmediate