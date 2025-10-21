from tkinter import W
from matplotlib.style import available
import pandas as pd
import numpy as np

def get_battery_details(window_length = 30):
    
    available = np.load(f"../data/processed/battery_schedule_window{window_length}.npy")
    available = [sample[~np.isnan(sample).any(axis=1)].astype(int) for sample in available]

    details = np.load(f"../data/processed/battery_details_window{window_length}.npy")
    
    index = np.load(f"../data/processed/battery_series_window{window_length}.npy")[:,0].astype(int)
    
    SOC_a_v, SOC_d_v, t_a_v, t_d_v = details[0], details[1], details[2], details[3]

    SOC_a_v = [sample[~np.isnan(sample)] for sample in SOC_a_v]
    SOC_d_v = [sample[~np.isnan(sample)] for sample in SOC_d_v]
    t_a_v = [sample[~np.isnan(sample)].astype(int) for sample in t_a_v]
    t_d_v = [sample[~np.isnan(sample)].astype(int) for sample in t_d_v]

    return index, available, SOC_a_v, SOC_d_v, t_a_v, t_d_v
