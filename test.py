# import numpy as np
# import config as cfg
# import os
# import pandas as pd
# from tqdm import tqdm
        

# base_dir = '/home/lisa4090/Documents/GitHub/V2B_Optimization_with_AI_on_BSS/data/battery_demand/tol25'
# case_dirs = {
#     1: os.path.join(base_dir, "case1_real_only"),
#     2: os.path.join(base_dir, "case2_nan_filled"),
#     3: os.path.join(base_dir, "case3_extended_generated"),
# }
# for d in case_dirs.values():
#     os.makedirs(d, exist_ok=True)

# # === Load raw data once ===
# df = pd.read_csv('data/battery_demand/tol25/resample_train.csv', index_col=0)
# df.index = pd.to_datetime(df.index)
# flat_values = df.values.flatten().tolist()

# # === Pass 1: classify windows ===
# clean_series = []
# eligible_chunks = []

# for i in range(len(flat_values) - cfg.WINDOW_LENGTH + 1):
#     if i == 9717 or i == 11093:
#         print(flat_values[i:i+cfg.WINDOW_LENGTH])
        
from time_series_generator.time_series_generator.core import Generator as BlockGenerator
from time_series_generator.time_series_generator.preprocessing import DataPrepare
import config as cfg
import numpy as np

sample_seed = [38.0, 44.0, 42.0, 46.0, 49.333333333333336, 50.5, 50.0, 47.0, 43.0, 20.0, 20.0, 30.0, np.nan, np.nan, np.nan, np.nan, 36.0, 38.5]

dataprep = DataPrepare(
    window_size=18,
    resolution=cfg.RESOLUTION,
    tolerance=1
)

# 新的呼叫方式
generator_random = BlockGenerator(
    seed=sample_seed,          
    datapreparer=dataprep,     
    window_size=18,
    max_shift=18,
    top_k=cfg.TOP_K,
    random_state=cfg.RANDOM_STATE,
)