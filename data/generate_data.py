import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import random

sys.path.append(os.path.abspath(".."))

from data.utils.preprocessor import DataPreprocessor
from data.utils.batterygenerator import BatterySeriesGenerator as gen

# Initialize the DataPreprocessor with the specified date range
print("Initializing DataPreprocessor...")
DataPreprocessor = DataPreprocessor(start_date="2023-01-01 00:00:00", end_date="2024-09-30 23:00:00")
print("DataPreprocessor initialized successfully.")

# Generate radiation data
print("Generating radiation data...")
radiation_data = DataPreprocessor.generate_radiation()
print("Radiation data generated successfully.")

# Generate temperature data
print("Generating temperature data...")
temperature_data = DataPreprocessor.generate_temperature()
print("Temperature data generated successfully.")

# Generate building data
print("Generating building data...")
building_data = DataPreprocessor.generate_building_data()
print("Building data generated successfully.")

# Generate electricity cost data for G2B and G2V scenarios
print("Generating electricity cost data...")
electricitycostG2B_data, electricitycostG2V_data = DataPreprocessor.generate_electricity_cost()
print("Electricity cost data generated successfully.")

# Define parameters for generating battery series
window_sizes = [36]  # Different window sizes to iterate over
tolerance = int(6)

gen = gen()

# Generate battery series data with and without NaN values for each window size
for window_size in window_sizes:
    print(f"Generating battery series data for window size {window_size}...")
    battery_series, battery_series_with_nan = gen.generate_battery_series(window_size=window_size, tolerance=tolerance, n_samples=5)
    print(f"Battery series data for window size {window_size} generated successfully. Shape: {battery_series.shape}, {battery_series_with_nan.shape}")

# Define parameters for generating battery schedules
n_station = 38 * 2  # Number of stations
SOC_thr = 0.9  # State of Charge threshold

# Generate battery schedules and details for each window size
for window_size in window_sizes:
    print(f"Generating battery schedule for window size {window_size}...")
    battery_schedule, battery_details = gen.generate_battery_schedule(n_station=n_station, SOC_thr=SOC_thr, window_size=window_size, tolerance=tolerance)
    print(f"Battery schedule for window size {window_size} generated successfully.")