import sys
import numpy as np
from collections import defaultdict
import os
from sklearn.neighbors import KernelDensity

import os, sys
from time_series_generator.time_series_generator.preprocessing  import DataPrepare
from time_series_generator.time_series_generator.metrics import fast_dtw_distance  # 仍可用在後驗細修時的輔助
from time_series_generator.time_series_generator.density import compute_posterior_weights_from_partial_subseq
import time_series_generator.time_series_generator.config as cfg

# ========= 規一化與相位對齊工具 =========

def _norm(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)                     # 忽略 NaN
    s = np.nanstd(x)                      # 忽略 NaN
    if not np.isfinite(s) or s < eps:
        s = eps                           # 避免除以 0 或極小值
    return np.nan_to_num((x - m) / s)     # 把 nan / inf 轉成有限值

def _best_lag_xcorr(seed, cand, max_shift):
    """
    以正規化互相關搜尋最佳位移。
    只在重疊且非 NaN 的位置計算。回傳 (best_lag, best_corr)
    """
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
    依 _best_lag_xcorr 的結果對時序平移。
    - mode="roll": 環狀平移（快、長度不變；對窗口資料可接受）
    - mode="crop": 取交集區間並以端點延伸補回原長度
    """
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if mode == "roll":
        return np.roll(a, -lag)  # 負號：讓樣本往「對齊 seed」方向平移
    elif mode == "crop":
        if lag >= 0:
            xs = a[lag:]
            out = np.empty_like(a)
            out[:n-lag] = xs
            out[n-lag:] = xs[-1]
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
    return np.sqrt(np.mean(d * d))


# ========= 生成器 =========

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
                 align_mode="roll",
                 bootstrap_size: int = cfg.BOOTSTRAP_SIZE if hasattr(cfg, "BOOTSTRAP_SIZE") else 20,
                 alpha_prior: float = getattr(cfg, "ALPHA_PRIOR", 2.0),
                 beta_lik: float = getattr(cfg, "BETA_LIK", 2.0),
                 ):
        self.window_size = window_size
        self.resolution = resolution
        self.seed = seed
        self.n_sample = n_sample
        self.bandwidth = bandwidth
        self.random_state = random_state
        self.max_shift = max_shift
        self.top_k = top_k
        self.align_mode = align_mode
        self.bootstrap_size = bootstrap_size
        self.alpha_prior = alpha_prior
        self.beta_lik = beta_lik

        self._estimator = BayesianDistributionEstimator(
            window_size=self.window_size,
            resolution=self.resolution,
            max_shift=self.max_shift,
            top_k=self.top_k,
            align_mode=self.align_mode,
            alpha_prior=self.alpha_prior,
            beta_lik=self.beta_lik
        )

    def generate(self):
        mean_seed, std_seed, X_joint, w_global, W_pos, mu_post, std_post = \
            self._estimator.estimate_and_correct_distribution_phase_locked(
                seed=self.seed, bandwidth=self.bandwidth
            )

        rng = np.random.RandomState(self.random_state)
        K, T = X_joint.shape
        h = self.bandwidth

        # Dirichlet 溫度：越小→越尖銳（越接近少數 analog）
        tau = 50.0
        alpha = np.clip(w_global, 1e-8, None) * tau

        samples = []
        for _ in range(self.n_sample):
            # 1) 以 Dirichlet 在 analog 上抽權重（不含 seed 值，因為 X_joint 來自歷史）
            a = rng.dirichlet(alpha)

            # 2) 形成「analog 先驗混合軌跡」：只在各樣本有觀測的位置累積
            obs_mask = np.isfinite(X_joint)                  # (K, T)
            X_filled = np.where(obs_mask, X_joint, 0.0)      # NaN 不貢獻
            # 位置別總權重，用來做歸一化（避免多 NaN 拉低）
            w_pos = np.sum((a[:, None] * obs_mask), axis=0)  # (T,)
            num = np.sum((a[:, None] * X_filled), axis=0)    # (T,)
            x_prior_mix = np.where(w_pos > 0, num / (w_pos + 1e-12), np.nan)

            # 3) 先驗混合上加 kernel 擾動（只在有值處）
            eps = rng.randn(T)
            x_prior_mix = np.where(np.isfinite(x_prior_mix),
                                x_prior_mix + eps * h,
                                np.nan)

            # 4) 後驗路徑（位置別 Gaussian）：完全不含 seed 值
            eta = rng.randn(T)
            x_post = mu_post + eta * np.maximum(std_post, 1e-12)

            # 5) 先驗/後驗混合係數：analog 較集中的情況，更多信任先驗
            #    這裡用 w_pos（位置上有效類比權重）做自適應
            gamma_t = 1.0 / (1.0 + w_pos)        # w_pos 大 → 倚賴先驗；小 → 倚賴後驗
            gamma_t = np.clip(gamma_t, 0.1, 0.9) # 合理界線

            # 6) 組合：有 analog 值處做混合；沒 analog 值處用後驗
            x_new = np.where(np.isfinite(x_prior_mix),
                            (1 - gamma_t) * x_prior_mix + gamma_t * x_post,
                            x_post)

            # 7) 不做「seed 反標準化」因為 X_joint/後驗都在原尺度
            samples.append(x_new)
        samples = np.stack(samples, axis=0)*std_seed + mean_seed
        return samples



# ========= 位置感知 + Bayesian 先驗/後驗的估計器 =========

class BayesianDistributionEstimator:
    """兩階段：先相位對齊選 Top-K，再在 Top-K 上做 Bayes 後驗修正（避免時間 shift）"""

    def __init__(self, window_size=cfg.WINDOW_SIZE, resolution=cfg.RESOLUTION,
                 max_shift=6, top_k=200, align_mode="roll",
                 alpha_prior: float = 2.0, beta_lik: float = 2.0):
        """
        alpha_prior: 放大相似樣本（先驗）的權重程度（越大越重視相似者）
        beta_lik:   放大不相似樣本（似然）的權重程度（越大越重視不相似者）
        """
        self.window_size = window_size
        self.resolution = resolution
        self.max_shift = max_shift
        self.top_k = top_k
        self.align_mode = align_mode
        self.alpha_prior = alpha_prior
        self.beta_lik = beta_lik

        self.datapreparer = DataPrepare(window_size, resolution)
        self.grouped_samples = None

    def estimate_and_correct_distribution_phase_locked(self, seed: np.ndarray, bandwidth=cfg.BANDWIDTH):
        """
        兩階段（Bayesian 位置感知版）：
        1) 針對所有候選樣本以正規化互相關搜尋最佳 lag，對齊後以「僅在重疊非 NaN 位置」的距離評分，取 Top-K
        2) 由距離→權重得到 w_global
        3) 先驗（相似樣本；alpha 放大） + 似然（不相似樣本；beta 放大）做位置別 Gaussian 共軛更新
        4) 得到後驗 mu_post, std_post

        回傳:
            mean_seed, std_seed                # seed 的標準化參數
            X_joint                            # (K, L) Top-K 對齊母體（可能含 NaN）
            w_global                           # (K,) 樣本等級的全域權重（由距離而來）
            W_pos                               # (K, L) 位置別權重（觀測處才有質量，僅供參考）
            mu_post, std_post                  # (L,) 位置別「後驗」均值與標準差
        """
        # ---------- 準備資料 ----------
        if self.grouped_samples is None:
            self._prepare_data()

        mean_seed, std_seed, normal_seed = self._normalize_seed(seed)

        keys = list(self.grouped_samples.keys())
        if not keys:
            raise ValueError("No grouped samples available.")
        keys = sorted(keys, key=lambda x: str(x))
        key2id = {k: i for i, k in enumerate(keys)}

        pool = []
        group_ids = []
        for k in keys:
            arr = np.asarray(self.grouped_samples[k], dtype=float)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[1] != self.window_size:
                continue
            pool.append(arr)
            group_ids.extend([key2id[k]] * arr.shape[0])

        if not pool:
            raise ValueError("No valid samples after length check; ensure subsequences have window_size length.")

        X_pool = np.vstack(pool)                # (N_total, L)
        group_ids = np.asarray(group_ids, int)  # (N_total,)
        L = X_pool.shape[1]

        # ---------- 階段 1：相位搜尋 + 對齊 + 以重疊距離評分，取 Top-K ----------
        def _norm_nan(x, eps=1e-8):
            x = np.asarray(x, dtype=float)
            m = np.nanmean(x)
            s = np.nanstd(x)
            if not np.isfinite(s) or s < eps:
                s = eps
            return (x - m) / s

        def _euclidean_overlap(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            if not np.any(mask):
                return 1e6
            d = a[mask] - b[mask]
            return float(np.sqrt(np.sum(d * d) / (np.sum(mask) + 1e-12)))

        N_total = X_pool.shape[0]
        N = min(50000, N_total)  # 可調上限以控時

        lags = np.zeros(N, dtype=int)
        scores = np.zeros(N, dtype=float)

        for i in range(N):
            lag, _ = _best_lag_xcorr(normal_seed, X_pool[i], self.max_shift)
            lags[i] = lag
            aligned = _apply_lag_roll(X_pool[i], lag, mode=self.align_mode)
            aligned_norm = _norm_nan(aligned)
            scores[i] = _euclidean_overlap(normal_seed, aligned_norm)

        k = min(self.top_k, N)
        top_idx = np.argsort(scores)[:k]

        # 形成已對齊的 Top-K 訓練母體（保留 NaN）
        X_joint = np.stack([_apply_lag_roll(X_pool[i], lags[i], mode=self.align_mode) for i in top_idx], axis=0)
        X_joint = np.asarray(X_joint, dtype=float)   # (K, L)
        K = X_joint.shape[0]

        # ---------- 與 seed 的樣本層級權重 w_global ----------
        d0 = np.empty(K, dtype=float)
        for i in range(K):
            d0[i] = _euclidean_overlap(normal_seed, _norm_nan(X_joint[i]))
        w_global = 1.0 / (d0**2 + 1e-8)
        s = w_global.sum()
        w_global = w_global / (s + 1e-12)

        # ---------- 位置別權重（供參考用；每個 t 歸一化） ----------
        obs_mask = np.isfinite(X_joint)             # (K, L)
        W_pos = w_global[:, None] * obs_mask        # (K, L) 未歸一化的「位置別權重」
        sum_w_pos = np.sum(W_pos, axis=0)           # (L,)
        valid_t = sum_w_pos > 0
        if np.any(valid_t):
            W_pos[:, valid_t] = W_pos[:, valid_t] / (sum_w_pos[valid_t][None, :])

        # ============================================================
        #   Bayes：先驗（相似樣本） + 似然（不相似樣本） → 後驗
        # ============================================================
        eps = 1e-12
        X_filled = np.where(obs_mask, X_joint, 0.0)   # 只在 obs_mask 位置用得到；0 不會被加權到

        # ---- 先驗：強化相似樣本 ----
        w_prior = w_global ** self.alpha_prior
        w_prior = w_prior / (w_prior.sum() + eps)

        Wp_pos = w_prior[:, None] * obs_mask
        sum_wp = np.sum(Wp_pos, axis=0)               # (L,)
        valid_p = sum_wp > 0

        mu_prior = np.full(L, np.nan, float)
        var_prior = np.full(L, np.nan, float)
        if np.any(valid_p):
            mu_prior[valid_p] = np.sum(Wp_pos[:, valid_p] * X_filled[:, valid_p], axis=0) / (sum_wp[valid_p] + eps)
            diffs_p = X_filled - mu_prior[None, :]
            var_prior[valid_p] = np.sum(Wp_pos[:, valid_p] * (diffs_p[:, valid_p] ** 2), axis=0) / (sum_wp[valid_p] + eps)

        std_prior = np.sqrt(np.maximum(var_prior, 0.0) + eps)

        # ---- 似然：強化不相似樣本 ----
        w_lik = (1.0 - w_global) ** self.beta_lik
        w_lik = w_lik / (w_lik.sum() + eps)

        Wl_pos = w_lik[:, None] * obs_mask
        sum_wl = np.sum(Wl_pos, axis=0)
        valid_l = sum_wl > 0

        mu_lik = np.full(L, np.nan, float)
        var_lik = np.full(L, np.nan, float)
        if np.any(valid_l):
            mu_lik[valid_l] = np.sum(Wl_pos[:, valid_l] * X_filled[:, valid_l], axis=0) / (sum_wl[valid_l] + eps)
            diffs_l = X_filled - mu_lik[None, :]
            var_lik[valid_l] = np.sum(Wl_pos[:, valid_l] * (diffs_l[:, valid_l] ** 2), axis=0) / (sum_wl[valid_l] + eps)

        std_lik = np.sqrt(np.maximum(var_lik, 0.0) + eps)

        # ---- 後驗：位置別 Gaussian 共軛更新 ----
        prec_prior = np.zeros(L, float)
        prec_lik = np.zeros(L, float)

        prec_prior[valid_p] = 1.0 / (std_prior[valid_p] ** 2 + eps)
        prec_lik[valid_l]   = 1.0 / (std_lik[valid_l] ** 2 + eps)

        prec_post = prec_prior + prec_lik
        var_post = np.full(L, np.nan, float)
        mu_post  = np.full(L, np.nan, float)
        valid_post = prec_post > 0

        if np.any(valid_post):
            var_post[valid_post] = 1.0 / (prec_post[valid_post] + eps)
            part_prior = np.zeros(L, float); part_prior[valid_p] = prec_prior[valid_p] * mu_prior[valid_p]
            part_lik   = np.zeros(L, float); part_lik[valid_l]   = prec_lik[valid_l]   * mu_lik[valid_l]
            mu_post[valid_post] = (part_prior[valid_post] + part_lik[valid_post]) * var_post[valid_post]

        std_post = np.sqrt(np.maximum(var_post, 0.0))

        # ---- 後備：若某些位置 prior/lik 都沒有，回退到群體共識（W_pos） ----
        mu_cons  = np.full(L, np.nan, float)
        var_cons = np.full(L, np.nan, float)
        if np.any(valid_t):
            mu_cons[valid_t] = np.sum(W_pos[:, valid_t] * X_filled[:, valid_t], axis=0)
            diffs_c = X_filled - mu_cons[None, :]
            var_cons[valid_t] = np.sum(W_pos[:, valid_t] * (diffs_c[:, valid_t] ** 2), axis=0)
        std_cons = np.sqrt(np.maximum(var_cons, 0.0))

        fallback = ~np.isfinite(mu_post)
        mu_post[fallback]  = mu_cons[fallback]
        std_post[fallback] = std_cons[fallback]

        # 極端保底
        mu_post[~np.isfinite(mu_post)]   = 0.0
        std_post[~np.isfinite(std_post)] = 1.0

        return mean_seed, std_seed, X_joint, w_global, W_pos, mu_post, std_post

    # ---------- DataPrepare 介面 ----------
    def _prepare_data(self):
        # 你的 DataPrepare 應提供 generate_grouped_subsequences()
        self.grouped_samples = self.datapreparer.generate_grouped_subsequences()

    # ---------- seed 的 z-score ----------
    def _normalize_seed(self, seed):
        seed = np.asarray(seed, dtype=float)
        mean_val = np.nanmean(seed)
        std_val  = np.nanstd(seed)
        if not np.isfinite(std_val) or std_val < 1e-12:
            std_val = 1.0  # 避免全常數造成除 0
        normal = (seed - mean_val) / std_val
        return mean_val, std_val, normal
