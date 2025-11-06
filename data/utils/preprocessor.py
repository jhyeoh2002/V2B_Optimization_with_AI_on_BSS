from operator import index
import os
import sys
import pandas as pd
import numpy as np


from scipy.stats import truncnorm
from datetime import datetime
from math import isnan

sys.path.append(os.path.abspath(".."))

from data.utils.webcrawler import get_building_data
from data.utils.costgenerator import get_building_cost, get_ev_cost
from data.utils.battscheduler import schedule_batteries
import time_series_generator.time_series_generator.core as tsg

class DataPreprocessor:
    
    # Add the project root directory to the Python module search path

    def __init__(self, start_date, end_date):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='h')
        
    def clean_weather_data(self, filepath):
        
        # Check if the file exists before reading
        if os.path.exists(filepath):
            data_df = pd.read_csv(filepath)
        else:
            print(f"File not found: {filepath}")
            return None

        # Drop unnecessary columns: '日/時' (date/hour) and '總和' (sum of radiation values)
        data_df.drop(['日/時'], axis=1, inplace=True)
        data_df.drop(columns=data_df.columns[-1], axis=1, inplace=True)

        # Transpose the DataFrame to make columns into rows
        data_df_transposed = data_df.T.reset_index()

        # Drop the last column after transposing, for the ‘總和‘ row
        data_df_transposed.drop(columns=data_df_transposed.columns[-1], axis=1, inplace=True)

        return data_df_transposed

    def generate_radiation(self):

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
                file_path = f'./raw/466920-{year}-{month:02d}-GlobalSolarRadiation-hour.csv'

                cleaned_df = self.clean_weather_data(file_path)
                # Reshape the DataFrame to have two columns: 'date' and 'Radiation'
                # 'index' column is used as an identifier for the reshaped data
                data_df_melted = cleaned_df.melt(id_vars=['index'], var_name='date', value_name='Radiation', ignore_index=True)
        
                # Concatenate the reshaped data for each month into the main DataFrame
                radiation_df = pd.concat([radiation_df, data_df_melted], ignore_index=True)

        # Add a 'Datetime' column using the pre-generated date range
        radiation_df['Datetime'] = self.date_range

        # Drop intermediate columns 'index' and 'date' as they are no longer needed
        radiation_df.drop(columns=['index', 'date'], axis=1, inplace=True)

        # Set 'Datetime' as the index for the final DataFrame
        radiation_df.set_index('Datetime', inplace=True)

        # Save the processed radiation data to a CSV file in the './processed/' directory
        # Ensure the directory exists before running this code
        radiation_df.to_csv('./processed/radiation_data.csv', index=True)

        return radiation_df


    def generate_temperature(self):

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
                file_path = f'./raw/466920-{year}-{month:02d}-AirTemperature-hour.csv'

                cleaned_df = self.clean_weather_data(file_path)
                # Reshape the DataFrame to have two columns: 'date' and 'Temperature'
                # 'index' column is used as an identifier for the reshaped data
                data_df_melted = cleaned_df.melt(id_vars=['index'], var_name='date', value_name='Temperature', ignore_index=True)

                # Concatenate the reshaped data for each month into the main DataFrame
                temperature_df = pd.concat([temperature_df, data_df_melted], ignore_index=True)

        # Add a 'Datetime' column using the pre-generated date range
        temperature_df['Datetime'] = self.date_range

        # Drop intermediate columns 'index' and 'date' as they are no longer needed
        temperature_df.drop(columns=['index', 'date'], axis=1, inplace=True)

        # Set 'Datetime' as the index for the final DataFrame
        temperature_df.set_index('Datetime', inplace=True)

        # Save the processed temperature data to a CSV file in the './processed/' directory
        # Ensure the directory exists before running this code
        temperature_df.to_csv('./processed/temperature_data.csv', index=True)
        
        return temperature_df
    
    def generate_building_data(self):

        complete_df = get_building_data(self.start_date.strftime("%Y-%m-%d %H:%M:%S"), self.end_date.strftime("%Y-%m-%d %H:%M:%S"))
        complete_df.to_csv('./raw/all_building.csv', index=False)

        building = complete_df.drop(columns=['時間/館舍'])
        building['Datetime'] = pd.to_datetime(building['datetime'])
        building = building.set_index('Datetime')

        building['-推廣中心-'] = pd.to_numeric(building['-推廣中心-'], errors='coerce')
        building['filled'] = building['-推廣中心-'].fillna(building['-推廣中心-'].shift(-24 * 7))
        building['Energy'] = building['filled'].fillna(building['filled'].shift(24 * 7))
        building['Energy'].to_csv('./processed/building_data.csv')
    
        return pd.DataFrame(building['Energy'])
    
    def generate_electricity_cost(self):

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

        cost_G2B_df.to_csv('./processed/electricitycostG2B_data.csv')
        cost_G2V_df.to_csv('./processed/electricitycostG2V_data.csv')

        return cost_G2B_df, cost_G2V_df

    def generate_battery_series(self, window_size = 48, tolerance = 6):

        fully_charged_df = pd.read_csv('../time_series_generator/modified_data/resample_data.csv',index_col=0)
        fully_charged_df.index = pd.to_datetime(fully_charged_df.index)
        
        flat_values = fully_charged_df.values.flatten().tolist()
        
        series = []
        series_with_nan = []

        for i in range(len(flat_values)-window_size):

            chunk = flat_values[i : i + window_size]
            nan_count = np.count_nonzero(np.isnan(chunk))
            
            if nan_count == 0:
                chunk.insert(0, i)
                series.append(chunk)
            
            elif nan_count <= tolerance and not (np.isnan(chunk[0]) or np.isnan(chunk[-1])):
                artificial_series = self.generate_artificial_battery_data(chunk)
                
                for artificial in artificial_series:
                    np.insert(artificial, 0, i)
                    series_with_nan.append(artificial)
                                
        series = np.array(series).astype(int)
        series_with_nan =  np.array(series_with_nan).astype(int)
        
        np.save(f'./processed/battery_series_with_nan_window{window_size}.npy', series_with_nan)        
        np.save(f'./processed/battery_series_window{window_size}.npy', series)

        return series, series_with_nan

    def generate_artificial_battery_data(self, list):

        generator = tsg.Generator(
            window_size=len(list),       # Length of each time series subsequence (default: 24)
            seed=list,                     # Input seed sequence (default: sampled from N(mean=40, std=20, size=window_size))
            n_sample=25              # Number of new samples to generate (default: 500)
        )
        
        artificial_series = generator.generate()
        # print(f"Seed Series: {np.array(list)}")
        
        # for i in range(len(artificial_series)):
            # print(f"Artificial Series {i+1}: {np.array(artificial_series[i]).astype(int)}")
            
        return artificial_series

    def generate_battery_schedule(self, n_station=38*2, SOC_thr=0.9, window_size=48):

        if os.path.isfile(f'./processed/battery_series_window{window_size}.npy'):
            battery_series = np.load(f'./processed/battery_series_window{window_size}.npy')
        else:
            battery_series = self.generate_battery_series(window_size=window_size)

        tnum = battery_series.shape[1] - 1  # Subtract 1 to account for the index column
        
        a_vt_list = []
        SOC_a_v_list = []
        SOC_d_v_list = []
        t_a_v_list = []
        t_d_v_list = []
        
        for series in battery_series:

            a_vt, SOC_a_v, SOC_d_v, t_a_v, t_d_v = schedule_batteries(series, n_station, tnum, SOC_thr=SOC_thr)
            a_vt_arr = np.array(a_vt, dtype=float)
            SOC_a_v_arr = np.round(np.array(SOC_a_v, dtype=float), 2)
            SOC_d_v_arr = np.round(np.array(SOC_d_v, dtype=float), 2)
            t_a_v_arr = np.array(t_a_v, dtype=float)
            t_d_v_arr = np.array(t_d_v, dtype=float)

            a_vt_list.append(a_vt_arr)
            SOC_a_v_list.append(SOC_a_v_arr)
            SOC_d_v_list.append(SOC_d_v_arr)
            t_a_v_list.append(t_a_v_arr)
            t_d_v_list.append(t_d_v_arr)

        # Find the maximum shape[0] across all arrays
        max_batt = max([arr.shape[0] for arr in a_vt_list])

        a_vt_list_arr = np.array([np.pad(arr, ((0, max_batt - arr.shape[0]), (0, 0)), mode='constant', constant_values=np.nan) for arr in a_vt_list])
        SOC_a_v_list_arr = np.array([np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan) for arr in SOC_a_v_list])
        SOC_d_v_list_arr = np.array([np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan) for arr in SOC_d_v_list])
        t_a_v_list_arr = np.array([np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan) for arr in t_a_v_list])
        t_d_v_list_arr = np.array([np.pad(arr, (0, max_batt - arr.shape[0]), mode='constant', constant_values=np.nan) for arr in t_d_v_list])

        details = np.array([SOC_a_v_list_arr, SOC_d_v_list_arr, t_a_v_list_arr, t_d_v_list_arr])
        
        np.save(f'./processed/battery_schedule_window{window_size}.npy', a_vt_list_arr)
        np.save(f'./processed/battery_details_window{window_size}.npy', details)

        return a_vt_list_arr, details