import numpy as np


A_PATH_TEMPLATE = '../Raw_Data/Gogoro/台北市大安區_臺大二活停車場站A ({:02d}).csv'
B_PATH_TEMPLATE = '../Raw_Data/Gogoro/台北市大安區_臺大二活停車場站B ({:02d}).csv'

RESOLUTION = '1h'
WINDOW_SIZE = 24

BANDWIDTH = 0.5

MEAN = 40
STD = 20
SEED = np.random.normal(loc=MEAN, scale=STD, size=WINDOW_SIZE)
NSAMPLE = 500

RANDOM_STATE = 42
