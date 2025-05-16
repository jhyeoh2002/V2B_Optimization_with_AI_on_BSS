import numpy as np
import pandas as pd

def create_time_index(timeend: np.datetime64, window_size: int) -> pd.DatetimeIndex:
    freq = '10min'
    return pd.date_range(end=timeend, periods=window_size, freq=freq)

def safe_nansum(arr, axis=None):
    result = np.nansum(arr, axis=axis)
    # 判斷原始陣列沿 axis 全部是 NaN 的位置
    all_nan = np.isnan(arr).all(axis=axis)
    
    # 將那些位置改為 np.nan
    if np.isscalar(result):
        return np.nan if all_nan else result
    result = result.astype('float64')  # 確保能裝 NaN
    result[all_nan] = np.nan
    return result