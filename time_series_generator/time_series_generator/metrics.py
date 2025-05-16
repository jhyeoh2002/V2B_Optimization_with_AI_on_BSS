import numpy as np
from dtaidistance import dtw

def fast_dtw_distance(ref, compare_arr):
    if len(compare_arr.shape) == 2:
        distances = np.empty(compare_arr.shape[0])
        for i in range(compare_arr.shape[0]):
            mask = ~np.isnan(compare_arr[i])
            distances[i] = dtw.distance(ref[mask], compare_arr[i][mask])
    else:
        mask = ~np.isnan(compare_arr)
        distances = dtw.distance(ref[mask], compare_arr[mask])
    return distances