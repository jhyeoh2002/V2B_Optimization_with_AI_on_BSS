from operator import index
import os
from platform import processor
import sys
import pandas as pd
import numpy as np
import config as cfg

from datetime import datetime
from math import isnan

from preprocess.webcrawler import get_building_data
from preprocess.costgenerator import get_building_cost, get_ev_cost
from preprocess.battscheduler import schedule_batteries
from preprocess.batterygenerator import BatterySeriesGenerator

class DataPreprocessor:
    
    # Add the project root directory to the Python module search path

    def __init__(self, start_date, end_date, tolerance):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='h')
        self.tolerance = tolerance
        
    def clean_weather_data(self, filepath):
        
        # Check if the file exists before reading
        if os.path.exists(filepath):
            data_df = pd.read_csv(filepath)
        else:
            print(f"\n\nFile not found: {filepath}")
            return None

        # Drop unnecessary columns: '日/時' (date/hour) and '總和' (sum of radiation values)
        data_df.drop(['日/時'], axis=1, inplace=True)
        data_df.drop(columns=data_df.columns[-1], axis=1, inplace=True)

        # Transpose the DataFrame to make columns into rows
        data_df_transposed = data_df.T.reset_index()

        # Drop the last column after transposing, for the ‘總和‘ row
        data_df_transposed.drop(columns=data_df_transposed.columns[-1], axis=1, inplace=True)

        return data_df_transposed

    def generate_radiation(self, filepath):

        # Initialize an empty DataFrame to store concatenated radiation data
        radiation_df = pd.DataFrame()

        for year in range(self.start_date.year, self.end_date.year + 1):
            for month in range(1, 13):

                # Skip months outside the specified date range
                if (year == self.start_date.year and month < self.start_date.month) or \
                   (year == self.end_date.year and month > self.end_date.month):
                    continue

                # Construct the file path dynamically based on year and month
                # Source: The raw data files are expected to be in the './raw/' directory
                file_path = f'./data/raw/466920-{year}-{month:02d}-GlobalSolarRadiation-hour.csv'

                cleaned_df = self.clean_weather_data(file_path)
                # Reshape the DataFrame to have two columns: 'date' and 'Radiation'
                # 'index' column is used as an identifier for the reshaped data
                data_df_melted = cleaned_df.melt(id_vars=['index'], var_name='date', value_name='Radiation', ignore_index=True)
        
                # Concatenate the reshaped data for each month into the main DataFrame
                radiation_df = pd.concat([radiation_df, data_df_melted], ignore_index=True)

        print("\t\t[INFO] Found radiation files from './data/raw/' in the format '466920-{year}-{month:02d}-GlobalSolarRadiation-hour.csv'")
        # Add a 'Datetime' column using the pre-generated date range
        radiation_df['Datetime'] = self.date_range

        # Drop intermediate columns 'index' and 'date' as they are no longer needed
        radiation_df.drop(columns=['index', 'date'], axis=1, inplace=True)

        # Set 'Datetime' as the index for the final DataFrame
        radiation_df.set_index('Datetime', inplace=True)

        # Save the processed radiation data to a CSV file in the './processed/' directory
        # Ensure the directory exists before running this code
        radiation_df.to_csv(filepath, index=True)
        print(f"\t\t[INFO] Processed radiation data saved to '{filepath}'")

        return radiation_df


    def generate_temperature(self, filepath):

        # Initialize an empty DataFrame to store concatenated temperature data
        temperature_df = pd.DataFrame()

        for year in range(self.start_date.year, self.end_date.year + 1):
            for month in range(1, 13):

                # Skip months outside the specified date range
                if (year == self.start_date.year and month < self.start_date.month) or \
                   (year == self.end_date.year and month > self.end_date.month):
                    continue

                # Construct the file path dynamically based on year and month
                # Source: The raw data files are expected to be in the './raw/' directory
                file_path = f'./data/raw/466920-{year}-{month:02d}-AirTemperature-hour.csv'

                cleaned_df = self.clean_weather_data(file_path)
                # Reshape the DataFrame to have two columns: 'date' and 'Temperature'
                # 'index' column is used as an identifier for the reshaped data
                data_df_melted = cleaned_df.melt(id_vars=['index'], var_name='date', value_name='Temperature', ignore_index=True)

                # Concatenate the reshaped data for each month into the main DataFrame
                temperature_df = pd.concat([temperature_df, data_df_melted], ignore_index=True)

        print("\t\t[INFO] Found temperature files from './data/raw/' in the format '466920-{year}-{month:02d}-AirTemperature-hour.csv'")
        
        # Add a 'Datetime' column using the pre-generated date range
        temperature_df['Datetime'] = self.date_range

        # Drop intermediate columns 'index' and 'date' as they are no longer needed
        temperature_df.drop(columns=['index', 'date'], axis=1, inplace=True)

        # Set 'Datetime' as the index for the final DataFrame
        temperature_df.set_index('Datetime', inplace=True)

        # Save the processed temperature data to a CSV file in the './processed/' directory
        # Ensure the directory exists before running this code
        temperature_df.to_csv(filepath, index=True)
        
        print(f"\t\t[INFO] Processed temperature data saved to '{filepath}'")
        
        return temperature_df
    
    def generate_building_data(self, filepath):

        complete_df = get_building_data(self.start_date.strftime("%Y-%m-%d %H:%M:%S"), self.end_date.strftime("%Y-%m-%d %H:%M:%S"))
        complete_df.to_csv('./data/raw/all_building.csv', index=False)

        building = complete_df.drop(columns=['時間/館舍'])
        building['Datetime'] = pd.to_datetime(building['datetime'])
        building = building.set_index('Datetime')

        building['-推廣中心-'] = pd.to_numeric(building['-推廣中心-'], errors='coerce')
        building['filled'] = building['-推廣中心-'].fillna(building['-推廣中心-'].shift(-24 * 7))
        building['Energy'] = building['filled'].fillna(building['filled'].shift(24 * 7))
        building['Energy'].to_csv(filepath)
        
        print(f"\t\t[INFO] Processed building data saved to '{filepath}'")
    
        return pd.DataFrame(building['Energy'])
    
    def generate_electricity_cost(self, filepath_G2B, filepath_G2V):
        
        print("\t\t[INFO] Generating electricity cost data...")

        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="d")

        chosen_dateli = []
        for chosen_date in date_range:
            chosen_dateli.append(chosen_date.strftime('%Y-%m-%d'))
            
        c_G2B_t = []
        c_G2V_t = []
        
        cost_G2B_df = pd.DataFrame()
        cost_G2V_df = pd.DataFrame()

        for chosen_date in chosen_dateli:
            c_G2B_t.extend(get_building_cost('twohigh', chosen_date))
            c_G2V_t.extend(get_ev_cost('evlow', chosen_date))
            
        cost_G2B_df['c_G2B_t'] = c_G2B_t
        cost_G2B_df['Datetime'] = self.date_range
        cost_G2V_df['c_G2V_t'] = c_G2V_t
        cost_G2V_df['Datetime'] = self.date_range
        
        cost_G2B_df = cost_G2B_df.set_index('Datetime')
        cost_G2V_df = cost_G2V_df.set_index('Datetime')

        cost_G2B_df.to_csv(filepath_G2B)
        cost_G2V_df.to_csv(filepath_G2V)
        
        print(f"\t\t[INFO] Processed electricity cost data saved to '{filepath_G2B}' and '{filepath_G2V}'")

        return cost_G2B_df, cost_G2V_df
    
    def preprocess(self):
        
        
        # Initialize the DataPreprocessor with the specified date range
        
        radiation_path = "./data/timeseries/radiation_data.csv"
        temperature_path = "./data/timeseries/temperature_data.csv"
        building_path = "./data/timeseries/building_data.csv"
        electricitycostG2B_path = "./data/timeseries/electricitycostG2B_data.csv"
        electricitycostG2V_path = "./data/timeseries/electricitycostG2V_data.csv"

        # Generate radiation data
        print("\n\t1.1 Preprocessing radiation data...")
        if not os.path.exists(radiation_path):
            self.generate_radiation(radiation_path)
        else:
            print(f"\t\t[INFO] Radiation data already exists at '{radiation_path}'")
        
        # Generate temperature data
        print("\n\t1.2 Preprocessing temperature data...")
        if not os.path.exists(temperature_path):
            self.generate_temperature(temperature_path)
        else:
            print(f"\t\t[INFO] Temperature data already exists at '{temperature_path}'")
        
        # Generate building data
        print("\n\t1.3 Preprocessing building data...")
        if not os.path.exists(building_path):
            self.generate_building_data(building_path)
        else:
            print(f"\t\t[INFO] Building data already exists at '{building_path}'")
        
        # Generate electricity cost data for G2B and G2V scenarios
        print("\n\t1.4 Preprocessing electricity cost data...")
        if not os.path.exists(electricitycostG2B_path) or not os.path.exists(electricitycostG2V_path):
            self.generate_electricity_cost(electricitycostG2B_path, electricitycostG2V_path)
        else:
            print(f"\t\t[INFO] Electricity cost data already exists at '{electricitycostG2B_path}' and '{electricitycostG2V_path}'")
        

        print(f"\n\t1.5 Preprocessing battery demand series data...")

        gen = BatterySeriesGenerator(self.tolerance)
        gen.generate_battery_series(rerun=False)


        print(f"\n\t1.6 Generating battery schedule...")
        gen.generate_battery_schedule(n_station=38 * 2, SOC_thr=cfg.SOC_THR, window_size=cfg.WINDOW_SIZE)

    