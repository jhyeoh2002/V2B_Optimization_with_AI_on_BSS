import numpy as np
from dtaidistance import dtw

# def fast_dtw_distance(ref, compare_arr):
#     if len(compare_arr.shape) == 2:
#         distances = np.empty(compare_arr.shape[0])
#         for i in range(compare_arr.shape[0]):
#             mask = ~np.isnan(compare_arr[i])
#             distances[i] = dtw.distance(ref[mask], compare_arr[i][mask])
#     else:
#         mask = ~np.isnan(compare_arr)
#         distances = dtw.distance(ref[mask], compare_arr[mask])
#     return distances


# def fast_dtw_distance(ref, compare_arr):
#     if len(compare_arr.shape) == 2:
#         distances = np.empty(compare_arr.shape[0])
#         for i in range(compare_arr.shape[0]):
#             mask = ~np.isnan(compare_arr[i])
#             distances[i] = np.linalg.norm(ref[mask] - compare_arr[i][mask])
#     else:
#         mask = ~np.isnan(compare_arr)
#         distances = np.linalg.norm(ref[mask] - compare_arr[mask])
#     return distances


def fast_dtw_distance(ref, compare_arr, max_shift=10):
    """
    計算 ref 與 compare_arr 之間的最小 DTW 距離，允許相位平移來達成最佳對齊。
    
    Parameters:
        ref: 1D numpy array, 參考訊號。
        compare_arr: 2D or 1D numpy array, 要比較的訊號集。
        max_shift: int, 最多允許多少點的相位平移（左右 shift 範圍）。
    
    Returns:
        distances: 若輸入為 2D，回傳每一列的最小 DTW 距離；若為 1D，回傳單一距離值。
    """
    def min_dtw_with_shift(a, b, max_shift):
        min_dist = float('inf')
        for shift in range(-max_shift, max_shift + 1):
            b_shifted = np.roll(b, shift)
            # 用 NaN mask 去除無效資料（如有）
            mask = ~np.isnan(a) & ~np.isnan(b_shifted)
            if np.sum(mask) < 2:  # 避免太少資料
                continue
            dist = dtw.distance(a[mask], b_shifted[mask])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    if len(compare_arr.shape) == 2:
        distances = np.empty(compare_arr.shape[0])
        for i in range(compare_arr.shape[0]):
            distances[i] = min_dtw_with_shift(ref, compare_arr[i], max_shift)
    else:
        distances = min_dtw_with_shift(ref, compare_arr, max_shift)

    return distances
