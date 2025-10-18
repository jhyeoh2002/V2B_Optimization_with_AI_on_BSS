import pandas as pd
import os
from Raw_Data.get_building_data import get_building_data
from Raw_Data.get_battery_info import generate_escooter_data
from Raw_Data.get_elec_cost import get_elec_cost

# Define the paths
gogoropath = './Raw_Data/Gogoro'
cleanedpath = './Cleaned_Data'

# Read the files from the raw data directory
filenames = os.listdir(gogoropath)
filenames.sort()

# Initialize DataFrames for each station
stationA = pd.DataFrame(columns=['time','batt_num'])
stationB = pd.DataFrame(columns=['time','batt_num'])

# Concatenate data from each station's files into the corresponding DataFrame
for f in filenames:
    if 'B' in f:
        raw = pd.read_csv(os.path.join(gogoropath, f))
        stationB = pd.concat([stationB, raw])
    elif 'A' in f:
        raw = pd.read_csv(os.path.join(gogoropath, f))
        stationA = pd.concat([stationA, raw])
    else:
        continue

# Ensure 'time' column is datetime and remove duplicates
stationA['time'] = pd.to_datetime(stationA['time'])
stationA = stationA.drop_duplicates(subset='time')
stationA.set_index('time', inplace=True)

stationB['time'] = pd.to_datetime(stationB['time'])
stationB = stationB.drop_duplicates(subset='time')
stationB.set_index('time', inplace=True)

# Resample both stations to hourly frequency, using nearest values within a limit
stationA_hourly = stationA.resample('h').nearest(limit=5)
stationB_hourly = stationB.resample('h').nearest(limit=5)

# Create a new DataFrame "Battery_Data" combining stationA and stationB columns
Battery_Data = pd.DataFrame({
    'datetime': stationA_hourly.index[:14960],
    'stationA': stationA_hourly['batt_num'].values[:14960],
    'stationB': stationB_hourly['batt_num'].values[:14960],
    'total': stationA_hourly['batt_num'].values[:14960] + stationB_hourly['batt_num'].values[:14960]
})

Battery_Data['Available'] = (Battery_Data['total'].fillna(Battery_Data['total'].shift(-24 * 7)) + Battery_Data['total'].fillna(Battery_Data['total'].shift(24 * 7)))/2

# Define start and end for slicing the data
batterystart, batteryend = 9432, 12360

# Save the complete Battery_Data to a CSV file
Battery_Data.to_csv('./Cleaned_Data/Battery_Data.csv', index=False)
Battery_Data.iloc[batterystart:batteryend].to_csv('./Cleaned_Data/Battery_Data_0129_0529.csv', index=False)

get_building_data(startdate="2024-01-29",  enddate="2024-5-29")
generate_escooter_data()
c_G2B_t, c_G2V_t = get_elec_cost(startdate="2024-01-01",  enddate="2024-12-31")

carbon = pd.read_csv('./Cleaned_Data/Carbon Intensity Data 2020.csv')
weather = pd.read_csv('./Cleaned_Data/Weather Data.csv')
battery = pd.read_csv('./Cleaned_Data/Battery_Data_0129_0529.csv')
building = pd.read_csv('./Cleaned_Data/building.csv')

date_times = pd.date_range(start="2024-01-01 00:00:00", end="2024-12-31 23:00:00", freq="h")

S_R_t = []
roof_area = 120 #m^2
pv_eff = 0.2036 #efficiency of PV cells

Rad = weather['Radiation, I'].tolist()
Temp = weather['Temp_average'].tolist()

for i in range(len(Rad)): ##Calculate generation by PV cells (kW)
    S_R_t.append(pv_eff*roof_area*Rad[i]*(1-0.005*(Temp[i]-25))/1000)

battery['datetime'] = pd.to_datetime(battery["datetime"])
battery.set_index('datetime')
battery = battery.drop(columns=['stationA','stationB','total'])

building['datetime'] = pd.to_datetime(building["datetime"])
building.set_index('datetime')

Full_data = pd.DataFrame({
    "datetime": date_times,
    "Carbon Intensity (kgC02eq/kWh)": carbon['kg CO2e'].tolist(), 
    "Radiation Intensity, I": weather['Radiation, I'].tolist(),
    "Temperature,T (deg)":weather['Temp_average'].tolist(),
    "PV Generation (kWh)":S_R_t,
    "Building Electricity Cost (NTD/kWh)":c_G2B_t,
    "Vehicle Electricity Cost (NTD/kWh)":c_G2V_t
    })

Full_data = Full_data.merge(battery, on="datetime", how="left", suffixes=('', '_new'))
Full_data = Full_data.merge(building, on="datetime", how="left", suffixes=('', '_new'))

Full_data[672:3600].to_csv('./Full_Data.csv', index=False)