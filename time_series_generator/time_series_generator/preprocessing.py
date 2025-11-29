import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Optional, Dict
from . import config as cfg
import os, sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add the parent directory

class DataPrepare:
    def __init__(self, window_size: int = 24, resolution: str = '1h', tolerance: int = None):
        self.window_size = window_size
        self.resolution = resolution
        self.tolerance = tolerance

    def generate_grouped_subsequences(
        self,
        a_path_template: str = cfg.A_PATH_TEMPLATE,
        b_path_template: str = cfg.B_PATH_TEMPLATE,
        a_range: range = range(40),
        b_range: range = range(37),
        remove_dates: List[str] = None     # <–– new parameter
        )  -> pd.Series:
        
        # # 1. Load data
        # a_tdf = self._load_series_from_range(a_path_template, a_range)
        # b_tdf = self._load_series_from_range(b_path_template, b_range)
        # tdf = pd.concat([a_tdf, b_tdf], axis=1).dropna().sum(axis=1)

        # # 2. Ensure minute-level continuity
        # full_index = pd.date_range(start=tdf.index.min(), end=tdf.index.max(), freq='1min')
        # tdf_filled = tdf.reindex(full_index).resample(self.resolution).mean()
        # tdf_filled.name = 'raw_data'

        # # save_path = "./data/battery_demand/resample_full.csv"
        # # tdf_filled.to_csv(save_path, index_label='timestamp')


        # # 3. Optional filtering before saving
        # if remove_dates:
        #     remove_dates = pd.to_datetime(remove_dates)
        #     tdf_filled = tdf_filled[~tdf_filled.index.normalize().isin(remove_dates)]
            
        # print(tdf_filled)
        
        # save_path = f"./data/battery_demand/tol{self.tolerance}/resample_train.csv"
        save_path = f'/home/gary/git_repo/jhern/V2B_Optimization_with_AI_on_BSS/data/battery_demand/tol1/resample_train.csv'
        tdf_filled = pd.read_csv(save_path, index_col=0, parse_dates=True)
        tdf_filled = tdf_filled["raw_data"]
    
        # print(tdf_filled)
                
        # 產生子序列並展開為固定長度
        subseqs = self._generate_valid_subsequences(tdf_filled.values)
        all_subseqs = self._expand_all_sequences(subseqs)
        normalized_subseqs = self._z_score_normalized(all_subseqs)
        
        # 按照數列位置分群
        grouped_samples = defaultdict(list)
        for i,row in enumerate(normalized_subseqs):
            not_nan_indices = np.where(~np.isnan(row))[0]
            if len(not_nan_indices) == 0:
                continue  # 跳過全 NaN 的 row

            # 觀測的位置
            observed = tuple(np.where(~np.isnan(row))[0])
            grouped_samples[observed].append(row)
            
        sorted_items = sorted(grouped_samples.items(), key=lambda x: len(x[0]), reverse=True)
        grouped_samples = dict(sorted_items)
        return grouped_samples

    def _load_series_from_range(self, path_template: str, index_range: range) -> pd.DataFrame:
        tdf = pd.DataFrame()
        for i in index_range:
            try:
                path = path_template.format(i)
                df = pd.read_csv(path, index_col=0)
                df.index = pd.to_datetime(df.index).floor('min')
                df = df[~df.index.duplicated()]
                tdf = pd.concat([tdf, df])
            except Exception:
                # print(f"Failed to load data from {path}")
                continue
        tdf = tdf[~tdf.index.duplicated()]
        tdf.sort_index(inplace=True)
        return tdf

    def _generate_valid_subsequences(self, arr: np.ndarray, min_len: int = 2) -> List[np.ndarray]:
        max_len = len(arr)
        subsequences = []
        N = len(arr)
        for i in range(N):
            for j in range(i + min_len, min(i + self.window_size, N) + 1):
                subseq = arr[i:j]
                if not np.isnan(subseq[0]) and not np.isnan(subseq[-1]):
                    subsequences.append(subseq)
        return subsequences

    def _expand_to_all_right_shifts(self, seq: np.ndarray) -> List[np.ndarray]:
        L = len(seq)
        assert L <= self.window_size, "序列長度不能超過 window_size"
        expanded_versions = []
        for shift in range(self.window_size - L + 1):
            left = np.full(shift, np.nan)
            right = np.full(self.window_size - L - shift, np.nan)
            padded = np.concatenate([left, seq, right])
            expanded_versions.append(padded)
        return expanded_versions

    def _expand_all_sequences(self, seq_list: List[np.ndarray]) -> np.ndarray:
        all_expanded = []
        for seq in seq_list:
            expanded = self._expand_to_all_right_shifts(seq)
            all_expanded.extend(expanded)
        return np.array(all_expanded)
    
    def _z_score_normalized(self,arr):
        # 計算 mean 與 std
        mean = np.nanmean(arr, axis=1, keepdims=True)
        std = np.nanstd(arr, axis=1, keepdims=True)

        # 建立空白陣列
        z_scaled = np.full_like(arr, np.nan)

        # 標準差 ≠ 0 的 row
        valid_rows = ((std != 0) & ~np.isnan(std))[:, 0]
        z_scaled[valid_rows] = (arr[valid_rows] - mean[valid_rows]) / std[valid_rows]

        # 標準差 = 0 的 row
        const_rows = ((std == 0) & ~np.isnan(std))[:, 0]
        z_scaled[const_rows] = np.where(
            ~np.isnan(arr[const_rows]),
            0.0,
            np.nan
        )
        return z_scaled
