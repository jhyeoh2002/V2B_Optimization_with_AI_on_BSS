from dtaidistance import dtw
import numpy as np

def fast_dtw_distance(ref, compare_arr, window_ratio=0.1, offdiag_penalty=0.0, lag_penalty=None):
    """
    window_ratio: Sakoe–Chiba 窗比例（越小越貼時序對齊）
    offdiag_penalty: 懲罰非對角步（>0 可減少時間拉伸）
    lag_penalty: 若給數值（例如 0.5），會加上 |lag| 懲罰；None 表示不加
    """
    def _core(a, b):
        # 1) 正確遮罩
        mask = ~np.isnan(a) & ~np.isnan(b)
        a = a[mask]; b = b[mask]
        if len(a) < 2 or len(b) < 2:
            return np.inf

        # 2) Sakoe–Chiba 窗 + 3) 非對角懲罰
        n = len(a)
        w = max(1, int(np.ceil(window_ratio * n)))
        d = dtw.distance(a, b, window=w, penalty=offdiag_penalty)

        # 4) 加上 lag 懲罰（可選）
        if lag_penalty is not None:
            # 簡易 lag 估計（也可換互相關）
            max_lag = max(1, n // 10)
            best_lag, best_score = 0, -np.inf
            for lag in range(-max_lag, max_lag+1):
                if lag < 0:
                    aa, bb = a[-lag:], b[:len(b)+lag]
                elif lag > 0:
                    aa, bb = a[:len(a)-lag], b[lag:]
                else:
                    aa, bb = a, b
                if len(aa) < 2:
                    continue
                s = np.corrcoef(aa, bb)[0,1]
                if np.isfinite(s) and s > best_score:
                    best_score, best_lag = s, lag
            d = d + lag_penalty * abs(best_lag)  # 或 (best_lag**2)

        return d

    if compare_arr.ndim == 2:
        distances = np.empty(compare_arr.shape[0])
        for i in range(compare_arr.shape[0]):
            distances[i] = _core(ref, compare_arr[i])
        return distances
    else:
        return _core(ref, compare_arr)
