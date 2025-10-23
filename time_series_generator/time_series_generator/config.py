import numpy as np


A_PATH_TEMPLATE = '../Raw_Data/Gogoro/台北市大安區_臺大二活停車場站A ({:02d}).csv'
B_PATH_TEMPLATE = '../Raw_Data/Gogoro/台北市大安區_臺大二活停車場站B ({:02d}).csv'

RESOLUTION = '1h'

BANDWIDTH = 0.05

MEAN = 40
STD = 8
SEED = np.array([31, 33, 30, 29, 21, 10, 15, 16, 16, 19, 13, 19, 12, 13, 17, 8, 9, 17, 18, 15, 19, 20, 25, 28])
SEED2 = np.array([37, 31, 28, 19, 28, 34, 38, 37, 40, 38, 29, 21, 34, 21, 15, 18, 28, 34, 37, 39, 42, 42, 43, 36])
WINDOW_SIZE = len(SEED)

# WINDOW_SIZE = 24
# SEED = np.random.normal(loc=MEAN, scale=STD, size=WINDOW_SIZE)

NSAMPLE = 10

RANDOM_STATE = 42
