import numpy as np


A_PATH_TEMPLATE = '../Raw_Data/Gogoro/台北市大安區_臺大二活停車場站A ({:02d}).csv'
B_PATH_TEMPLATE = '../Raw_Data/Gogoro/台北市大安區_臺大二活停車場站B ({:02d}).csv'

RESOLUTION = '1h'
WINDOW_SIZE = 50

BANDWIDTH = 0.5

MEAN = 40
STD = 5
# SEED = np.array([30, 36, 43, 46, 44, 44, 45, 46, 26, 24, 34, 41, 41, 40, 46, 45, 53, 38, 32, 32, 35, 32, 25, 22])
SEED = np.random.normal(loc=MEAN, scale=STD, size=WINDOW_SIZE)
NSAMPLE = 50

RANDOM_STATE = 42
