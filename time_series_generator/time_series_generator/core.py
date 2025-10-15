import numpy as np
from collections import defaultdict
from sklearn.neighbors import KernelDensity
from time_series_generator.preprocessing  import DataPrepare
from time_series_generator.metrics import fast_dtw_distance  # 仍可用在後驗細修時的輔助
from time_series_generator.density import compute_posterior_weights_from_partial_subseq
import time_series_generator.config as cfg

# ========= 新增：相位對齊工具 =========

def _norm(x):
    x = np.asarray(x, dtype=float)
    m, s = np.mean(x), np.std(x)
    return (x - m) / (s + 1e-12)

def _best_lag_xcorr(seed, cand, max_shift):
    x = _norm(seed).ravel()
    y = _norm(cand).ravel()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    best_corr, best_lag = -np.inf, 0
    for lag in range(-max_shift, max_shift + 1):
        if lag >= 0:
            xs, ys = x[lag:], y[:n - lag]
        else:
            xs, ys = x[:n + lag], y[-lag:]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() < 2:
            continue
        a, b = xs[mask], ys[mask]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        c = float(np.dot(a, b) / denom)
        if c > best_corr:
            best_corr, best_lag = c, lag
    return best_lag, best_corr

def _apply_lag_roll(arr, lag, mode="roll"):
    """
    對候選序列套用 lag 來對齊 seed。
    - mode="roll": 環狀平移（快、長度不變；對週期性片段通常可接受）
    - mode="crop": 依 lag 取交集區間並回填到原長度（邊界以端點延伸）
    """
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if mode == "roll":
        return np.roll(a, -lag)  # 注意：負號讓 y 往「對齊 seed」方向平移
    elif mode == "crop":
        if lag >= 0:
            xs = a[lag:]
            out = np.empty_like(a)
            out[:n-lag] = xs
            out[n-lag:] = xs[-1]  # 端點延伸
        else:
            xs = a[:n+lag]
            out = np.empty_like(a)
            out[-(n+lag):] = xs
            out[:-(n+lag)] = xs[0]
        return out
    else:
        raise ValueError("mode must be 'roll' or 'crop'")
    
def _euclidean(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.inf
    d = a[mask] - b[mask]
    return np.sqrt(np.mean(d*d))


class Generator:
    def __init__(self,
                 window_size=cfg.WINDOW_SIZE,
                 resolution=cfg.RESOLUTION,
                 seed=cfg.SEED,
                 n_sample=cfg.NSAMPLE,
                 bandwidth=cfg.BANDWIDTH,
                 random_state=cfg.RANDOM_STATE,
                 max_shift=cfg.MAX_SHIFT if hasattr(cfg, "MAX_SHIFT") else 6,
                 top_k=cfg.TOP_K if hasattr(cfg, "TOP_K") else 200,
                 align_mode="roll"):
        self.window_size = window_size
        self.resolution = resolution
        self.seed = seed
        self.n_sample = n_sample
        self.bandwidth = bandwidth
        self.random_state = random_state
        self.max_shift = max_shift
        self.top_k = top_k
        self.align_mode = align_mode

        self._estimator = BayesianDistributionEstimator(
            window_size=self.window_size,
            resolution=self.resolution,
            max_shift=self.max_shift,
            top_k=self.top_k,
            align_mode=self.align_mode
        )

    def generate(self):
        mean_seed, std_seed, X_joint, w_post = self._estimator.estimate_and_correct_distribution_phase_locked(
            seed=self.seed, bandwidth=self.bandwidth
        )

        kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
       # ---- 先挑出沒有 NaN/Inf 的樣本列 ----
        row_ok = np.all(np.isfinite(X_joint), axis=1)
        X_kde = X_joint[row_ok]
        w_kde = w_post[row_ok]

        # ---- 權重清理（把 NaN/Inf/負值 → 0）----
        w_kde = np.nan_to_num(w_kde, nan=0.0, posinf=0.0, neginf=0.0)
        w_kde = np.clip(w_kde, 0.0, None)

        # 若仍全為 0，則退回均勻/常數補值策略
        if X_kde.shape[0] == 0 or float(w_kde.sum()) == 0.0:
            # 退而求其次：用補 0 的 X 與均勻權重
            X_kde = np.nan_to_num(X_joint, nan=0.0, posinf=0.0, neginf=0.0)
            w_kde = np.ones(X_kde.shape[0], dtype=float)

        # ---- 權重正規化（非必要但穩定）----
        w_sum = float(w_kde.sum())
        if w_sum <= 0:
            w_kde[:] = 1.0 / len(w_kde)
        else:
            w_kde = w_kde / w_sum

        # ---- 最終防呆 ----
        assert np.isfinite(X_kde).all(), "X_kde 仍含 NaN/Inf"
        assert np.isfinite(w_kde).all(), "w_kde 仍含 NaN/Inf"
        assert X_kde.shape[0] == w_kde.shape[0], "樣本數與權重數不一致"

        # ---- 擬合 KDE ----
        kde.fit(X_kde, sample_weight=w_kde)

        new_samples = kde.sample(n_samples=self.n_sample, random_state=self.random_state)
        new_samples = new_samples * std_seed + mean_seed
        return new_samples


class BayesianDistributionEstimator:
    """兩階段：先相位對齊選 Top-K，再在 Top-K 上做後驗修正（避免時間 shift）"""

    def __init__(self, window_size=cfg.WINDOW_SIZE, resolution=cfg.RESOLUTION,
                 max_shift=6, top_k=200, align_mode="roll"):
        self.window_size = window_size
        self.resolution = resolution
        self.max_shift = max_shift
        self.top_k = top_k
        self.align_mode = align_mode
        self.datapreparer = DataPrepare(window_size, resolution)
        self.grouped_samples = None

    def estimate_and_correct_distribution_phase_locked(self, seed: np.ndarray, bandwidth=cfg.BANDWIDTH):
        """
        兩階段：
        1) 針對所有候選樣本以正規化互相關搜尋最佳 lag，對齊後用 Euclidean 排名取 Top-K
        2) 僅在已對齊的 Top-K 上進行後驗微調（不再允許時間彈性對齊）
        回傳: mean_seed, std_seed, X_joint(Top-K 對齊母體), w_post(後驗權重)
        """
        if self.grouped_samples is None:
            self._prepare_data()

        mean_seed, std_seed, normal_seed = self._normalize_seed(seed)

        # ---- 收集所有群組的樣本（母體池）----
        keys = list(self.grouped_samples.keys())
        if not keys:
            raise ValueError("No grouped samples available.")

        # 穩定排序避免每次順序不同；key 可能是 tuple/自定義型別，轉字串排序較保險
        keys = sorted(keys, key=lambda x: str(x))
        key2id = {k: i for i, k in enumerate(keys)}

        pool = []
        group_ids = []  # 每筆樣本對應的整數群組 ID
        for k in keys:
            arr = np.asarray(self.grouped_samples[k], dtype=float)
            # 轉成 (n, L)
            if arr.ndim == 1:
                arr = arr[None, :]
            # 長度防呆：確保與 window_size 一致（不足可跳過或補齊；這裡採跳過）
            if arr.shape[1] != self.window_size:
                # 也可改成 arr = arr[:, :self.window_size] 進行裁切
                continue
            pool.append(arr)
            group_ids.extend([key2id[k]] * arr.shape[0])

        if not pool:
            raise ValueError("No valid samples after length check; ensure subsequences have window_size length.")

        X_pool = np.vstack(pool)               # (N_total, window_size)
        group_ids = np.asarray(group_ids, int) # (N_total,)

        # ---- 階段 1：相位搜尋 + 對齊 + 以 Euclidean 排名，取 Top-K ----
        N = X_pool.shape[0]
        lags = np.zeros(N, dtype=int)
        scores = np.zeros(N, dtype=float)

        # 對所有樣本搜尋最佳 lag（限制在 ±max_shift）
        for i in range(N):
            lag, _ = _best_lag_xcorr(normal_seed, X_pool[i], self.max_shift)
            lags[i] = lag
            aligned = _apply_lag_roll(X_pool[i], lag, mode=self.align_mode)
            # 對齊後用 Euclidean（不允許再時間扭曲）評分；先做標準化避免幅度影響
            scores[i] = _euclidean(normal_seed, _norm(aligned))

        k = min(self.top_k, N)
        top_idx = np.argsort(scores)[:k]

        # 形成已對齊的 Top-K 訓練母體
        X_joint = np.stack([_apply_lag_roll(X_pool[i], lags[i], mode=self.align_mode) for i in top_idx], axis=0)
        X_joint = np.asarray(X_joint, dtype=float)

        # 初始先驗（以對齊後距離當作權重）
        d0 = np.array([_euclidean(normal_seed, _norm(X_joint[i])) for i in range(X_joint.shape[0])], dtype=float)
        w_prior = 1.0 / (d0**2 + 1e-8)
        w_prior_sum = w_prior.sum()
        w_prior = w_prior / (w_prior_sum if w_prior_sum > 0 else 1.0)
        w_post = w_prior.copy()

        # ---- 階段 2：在 Top-K 上做群組視角的後驗微調（不再允許大位移）----
        top_groups = group_ids[top_idx]
        for k_key in keys:
            gid = key2id[k_key]
            mask = (top_groups == gid)
            if not np.any(mask):
                continue

            X_obs = X_joint[mask]  # 該群在 Top-K 的已對齊樣本
            # 以與 seed 的距離（已對齊）產生觀測權重
            d_obs = np.array([_euclidean(normal_seed, _norm(x)) for x in X_obs], dtype=float)
            w_obs = 1.0 / (d_obs**2 + 1e-8)
            w_obs_sum = w_obs.sum()
            w_obs = w_obs / (w_obs_sum if w_obs_sum > 0 else 1.0)

            # 用既有的部分序列後驗修正函式做細部修正（支撐集固定為 X_joint）
            w_post = compute_posterior_weights_from_partial_subseq(
                X_joint, w_post, X_obs, w_obs, bandwidth=bandwidth
            )

        return mean_seed, std_seed, X_joint, w_post


    def _prepare_data(self):
        self.grouped_samples = self.datapreparer.generate_grouped_subsequences()

    def _normalize_seed(self, seed):
        mean_val = np.mean(seed)
        std_val = np.std(seed)
        if np.allclose(seed, seed[0]):
            normal = seed / (mean_val + 1e-12) if mean_val != 0 else np.zeros_like(seed)
        else:
            normal = (seed - mean_val) / (std_val + 1e-12)
        return mean_val, std_val, normal
