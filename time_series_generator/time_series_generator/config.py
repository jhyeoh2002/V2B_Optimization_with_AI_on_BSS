import numpy as np
import os 

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
raw_data_dir = os.path.join(base_dir, "Raw_Data", "Gogoro")

A_PATH_TEMPLATE = os.path.join(raw_data_dir, '台北市大安區_臺大二活停車場站A ({:02d}).csv')
B_PATH_TEMPLATE = os.path.join(raw_data_dir, '台北市大安區_臺大二活停車場站B ({:02d}).csv')

RESOLUTION = '1h'

BANDWIDTH = 0.001

MEAN = 27.91
STD = 10.34
RG_WINDOW_SIZE = 30
RG_SEED = np.random.normal(loc=MEAN, scale=STD, size=RG_WINDOW_SIZE)

# window size = 24
SEED = np.array([31, 33, 30, 29, 21, 10, 15, 16, 16, 19, 13, 19, 12, 13, 17, 8, 9, 17, 18, 15, 19, 20, 25, 28])
SEED2 = np.array([37, 31, 28, 19, 28, 34, 38, 37, 40, 38, 29, 21, 34, 21, 15, 18, 28, 34, 37, 39, 42, 42, 43, 36])
WINDOW_SIZE = len(SEED)

# window size = 48
# SEED_48 = np.array([3.0, 18.0, 21.0, 25.0, 15.4, 20.33333333, 28.6, 30.5, 35.66666667, 37.0, 37.0, 36.5, 33.5, 26.33333333, 10.33333333, 6.0, 10.0, 14.6, 19.0, 18.0, 15.66666667, 20.66666667, 26.0, 15.28571429, 9.0, 16.33333333, 21.25, 20.0, 8.0, 12.5, 15.0, 24.33333333, 31.5, 35.0, 32.0, 35.0, 34.6, 32.0, 31.6, 30.0, 22.66666667, 14.5, 18.8, 27.0, 18.6, 19.2, 19.66666667, 25.25])
WINDOW_SIZE_48 = 48
SEED_48 = np.array([28.0, 26.0, 22.0, np.nan, np.nan, 21.0, 28.0, 17.0, 32.0, 36.0, 30.0, 19.0, 20.0, np.nan, np.nan, 33.0, 37.0, 42.0, 42.0, 42.0, 38.67, 33.33, 30.8, 27.33, 29.2, 30.0, 29.6, 26.0, np.nan, 26.67, 16.0, 21.0, 20.67, 25.5, 26.0, 22.0, 16.33, 22.0, 32.0, 35.0, 40.0, 40.0, 40.0, 38.0, np.nan, 6.0, 13.0, 19.33])
# SEED_48 = np.array([3.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 37.0, 37.0, 36.5, 33.5, 26.33333333, 10.33333333, 6.0, 10.0, 14.6, 19.0, 18.0, 15.66666667, 20.66666667, 26.0, 15.28571429, 9.0, 16.33333333, 21.25, 20.0, 8.0, 12.5, 15.0, 24.33333333, 31.5, 35.0, 32.0, 35.0, 34.6, 32.0, 31.6, 30.0, 22.66666667, 14.5, 18.8, 27.0, 18.6, 19.2, 19.66666667, 25.25])

# SEED_48 = np.random.normal(loc=MEAN, scale=STD, size=WINDOW_SIZE_48)
TOP_K = 200

BOOTSTRAP_SIZE = 1
NSAMPLE = 25
RANDOM_STATE = 42
MAX_SHIFT = 24