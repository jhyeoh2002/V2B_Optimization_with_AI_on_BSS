import numpy as np


A_PATH_TEMPLATE = '../Raw_Data/Gogoro/台北市大安區_臺大二活停車場站A ({:02d}).csv'
B_PATH_TEMPLATE = '../Raw_Data/Gogoro/台北市大安區_臺大二活停車場站B ({:02d}).csv'

RESOLUTION = '1h'
# WINDOW_SIZE = 24

BANDWIDTH = 0.5

MEAN = 40
STD = 8
SEED = np.array([46, 45, 44, 45, 44, 48, 49, 50, 46, 44, 41, 32, 39, 40, 36, 38, 37, 37, 43, 45, 38, 35, 31, 38])
WINDOW_SIZE = len(SEED)

# SEED = np.random.normal(loc=MEAN, scale=STD, size=WINDOW_SIZE)
NSAMPLE = 5000

RANDOM_STATE = 42
