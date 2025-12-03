import time
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


# =============================================================================
# 0. 基礎工具 (Base Utilities)
# =============================================================================

def _norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    計算 Z-Score 標準化 (Standardization)。
    將序列轉換為零均值、單位標準差。（以自身 mean/std）
    """
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)

    if not np.isfinite(s) or s < eps:
        s = eps

    return (x - m) / s


def _best_lag_xcorr(seed: np.ndarray, cand: np.ndarray, max_shift: int) -> np.ndarray:
    """
    尋找最佳的延遲 (Lag) 以對齊 Seed 和 Candidate，並返回對齊後的 Candidate。
    基於最大 Pearson 互相關係數 (Cross-Correlation) 原則。
    """
    x = _norm(seed)
    y = _norm(cand)

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    best_corr, best_lag = -np.inf, 0
    std_eps = 1e-8

    for lag in range(-max_shift, max_shift + 1):
        if lag >= 0:
            xs, ys = x[lag:], y[:n - lag]
        else:
            xs, ys = x[:n + lag], y[-lag:]

        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() < 2:
            continue

        a = xs[mask]
        b = ys[mask]

        if np.std(a) < std_eps or np.std(b) < std_eps:
            corr = -1.0
        else:
            corr = np.corrcoef(a, b)[0, 1]

        if np.isnan(corr):
            corr = -1.0

        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return np.roll(np.asarray(cand, float), -best_lag)


def _euclidean_distance(seed_norm: np.ndarray, cand_norm: np.ndarray) -> float:
    """
    計算歐式距離 (RMSE) 作為相似度指標。
    """
    mask = np.isfinite(seed_norm) & np.isfinite(cand_norm)
    if mask.sum() == 0:
        return np.inf

    diff = seed_norm[mask] - cand_norm[mask]
    return np.sqrt(np.mean(diff ** 2))


# =============================================================================
# 1. 貝氏估計器 (Bayesian Estimator) - 機率重抽樣核心
# =============================================================================

class BayesianEstimator:
    """
    用於計算歷史片段 (samples) 相對於給定 seed 的相似度機率，並進行重抽樣。
    """

    def __init__(
        self,
        samples: List[np.ndarray],
        window_size: int,
        max_shift: int,
        top_k: int,
    ) -> None:
        self.samples = samples
        self.window_size = window_size
        self.max_shift = max_shift
        self.top_k = top_k
        # 可以選擇性掛一個 random_state 在這裡，現在是 lazy 初始化
        self.random_state = None

    def get_resampled_pool(
        self,
        seed: np.ndarray,
        n_draws: int = 100,
        selection_mode: str = "topk",
        top_k_limit: Union[int, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算所有歷史片段相對 seed 的相似度，並重抽樣出 n_draws 條樣本。

        Returns
        -------
        resampled_pool : (n_draws, T)
        resampled_weights : (n_draws,)
        """
        # ---- 0. 準備資料形狀 ----
        seed = np.asarray(seed, float)  # (T,)
        seed_norm = _norm(seed)        # (T,)

        samples_arr = np.asarray(self.samples, float)  # (N, T)
        if samples_arr.ndim != 2:
            raise ValueError("self.samples 必須是形狀為 (N, T) 的 2D 陣列")

        N, T = samples_arr.shape

        if seed.shape[0] != T:
            raise ValueError(
                f"seed 長度 {seed.shape[0]} 與 samples 長度 {T} 不一致"
            )

        max_shift = self.max_shift

        # ---- 1. 一次性 normalize 所有樣本 ----
        eps = 1e-8
        mean = np.nanmean(samples_arr, axis=1, keepdims=True)  # (N, 1)
        std = np.nanstd(samples_arr, axis=1, keepdims=True)    # (N, 1)
        std = np.where((~np.isfinite(std)) | (std < eps), 1.0, std)

        samples_norm = (samples_arr - mean) / std  # (N, T)

        # ---- 2. 掃過所有 lag，找每條樣本的最佳 lag ----
        best_d2 = np.full(N, np.inf, dtype=float)
        best_lag = np.zeros(N, dtype=int)

        seed_norm_2d = seed_norm[None, :]  # (1, T)

        for lag in range(-max_shift, max_shift + 1):
            shifted_norm = np.roll(samples_norm, shift=lag, axis=1)  # (N, T)
            diff = shifted_norm - seed_norm_2d
            d2 = np.nansum(diff * diff, axis=1)  # (N,)

            mask = d2 < best_d2
            best_d2[mask] = d2[mask]
            best_lag[mask] = lag

        distances = np.sqrt(best_d2)  # (N,)

        # ---- 3. 依據 best_lag 對原始 samples 做對齊 ----
        aligned_candidates = np.empty_like(samples_arr)  # (N, T)
        unique_lags = np.unique(best_lag)

        for lag in unique_lags:
            idx = (best_lag == lag)
            aligned_candidates[idx] = np.roll(samples_arr[idx], shift=lag, axis=1)

        # ---- 4. 根據模式計算機率 ----
        # selection_mode = selection_mode.lower()
        
        # ---- 4. 根據模式計算機率 (Modified for Weighted Top-K) ----
        selection_mode = selection_mode.lower()

        if selection_mode == "topk":
            K = min(top_k_limit if top_k_limit is not None else self.top_k, N)
            best_indices = np.argsort(distances)[:K]

            probs = np.zeros(N, dtype=float)
            if K > 0:
                # [修改重點]：原本是均等權重 (1.0/K)，現在改為距離倒數權重
                # 取出前 K 名的距離
                top_dists = distances[best_indices]
                
                # 避免除以零
                epsilon = 1e-6
                
                # 使用 Power Law (冪次律) 來放大差異
                # power 越大，越極端地偏好第一名
                # power = 5 是原本 fallback 的設定，已經很強了；若要更強可設 10
                power = 10
                
                weights = 1.0 / (top_dists ** power + epsilon)
                
                # 正規化成機率
                sub_probs = weights / np.sum(weights)
                
                # 填回總機率表
                probs[best_indices] = sub_probs
                
            else:
                # 萬一 K == 0 就 fallback
                epsilon = 1e-6
                weights = 1.0 / (distances ** power + epsilon)
                probs = weights / np.sum(weights)
                
        # if selection_mode == "topk":
        #     K = min(top_k_limit if top_k_limit is not None else self.top_k, N)
        #     best_indices = np.argsort(distances)[:K]

        #     probs = np.zeros(N, dtype=float)
        #     if K > 0:
        #         probs[best_indices] = 1.0 / K
        #         probs /= np.sum(probs)
        #     else:
        #         # 萬一 K == 0 就 fallback 成 soft weighting
        #         epsilon = 1e-6
        #         weights = 1.0 / (distances ** 2 + epsilon)
        #         probs = weights / np.sum(weights)
        # else:
        #     # "soft" 模式：距離越小權重越大
        #     epsilon = 1e-6
        #     weights = 1.0 / (distances ** 2 + epsilon)
        #     probs = weights / np.sum(weights)

        # ---- 5. 重抽樣 ----
        rng = self.random_state
        if rng is None:
            rng = np.random.default_rng()

        indices = rng.choice(N, size=n_draws, p=probs, replace=True)
        resampled_pool = aligned_candidates[indices]  # (n_draws, T)
        resampled_weights = probs[indices]           # (n_draws,)

        return resampled_pool, resampled_weights


# =============================================================================
# 2. 混合生成器 (Mixture Generator) - 結合與加噪
# =============================================================================

class BayesianMixtureGenerator:
    """
    將重抽樣的歷史片段與 seed 混合，並加入加權觀測雜訊。

    這個版本的「往 seed 靠的強度」 alpha_i 是 data-driven，
    由 seed 的變異與 sample 與 seed 的差異變異自動決定。
    """

    def __init__(
        self,
        seed: np.ndarray,
        base_noise: float = 0.1,
        random_state: int = 0,
    ) -> None:
        self.seed = np.asarray(seed, float)
        self.T = len(seed)
        self.base_noise = float(base_noise)
        self.rng = np.random.RandomState(random_state)

        # 預先計算 seed 在有效區域的變異 (作為 s^2_seed)
        seed_mask = np.isfinite(self.seed)
        if seed_mask.sum() >= 2:
            self.seed_var = float(np.nanvar(self.seed[seed_mask]))
        else:
            self.seed_var = 1.0  # fallback，避免 0 或 NaN

        if (not np.isfinite(self.seed_var)) or (self.seed_var <= 1e-8):
            self.seed_var = 1.0

    def _compute_alpha(
        self,
        sample: np.ndarray,
        overlap_mask: np.ndarray,
        eps: float = 1e-8,
    ) -> float:
        """
        根據「sample 與 seed 在 overlap 區域的差異變異」自動計算 alpha_i。

        alpha_i = s^2_seed / (s^2_seed + s^2_diff)

        - s^2_diff 小 → sample 已經很像 seed → alpha_i 接近 1（少修）
        - s^2_diff 大 → sample 跟 seed 差很多 → alpha_i 接近 0（多往 seed 拉）
        """
        if overlap_mask.sum() < 2:
            # overlap 太少，直接不動 sample（alpha = 1）
            return 1.0

        diff = sample[overlap_mask] - self.seed[overlap_mask]
        s2_diff = float(np.nanvar(diff))

        if (not np.isfinite(s2_diff)) or (s2_diff < 0.0):
            s2_diff = 0.0

        s2_seed = self.seed_var

        alpha = s2_seed / (s2_seed + s2_diff + eps)
        alpha = float(np.clip(alpha, 0.0, 1.0))  # clamp 避免極端值

        return alpha

    def generate_raw_ensemble(
        self,
        pool: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        直接使用 Resampled Pool 進行混合，產生 Raw Ensemble。

        Parameters
        ----------
        pool : np.ndarray
            對齊後的重抽樣樣本池，shape = (n_sample, T)。
        weights : np.ndarray
            每個樣本在原始分佈中被選中的機率，shape = (n_sample,)。

        Returns
        -------
        samples : np.ndarray
            混合後的原始樣本集 (Raw Ensemble)，shape = (n_sample, T)。
        """
        pool = np.asarray(pool, float)
        weights = np.asarray(weights, float)

        n_sample, T = pool.shape
        if T != self.T:
            raise ValueError(f"pool 的長度 {T} 與 seed 的長度 {self.T} 不一致")

        samples = []
        seed_mask = np.isfinite(self.seed)

        for i in range(n_sample):
            sample = np.copy(pool[i])

            # 1. Level shift：先做均值平移，對齊平均高度
            cand_mask = np.isfinite(sample)
            overlap = seed_mask & cand_mask

            if overlap.sum() > 0:
                diff = self.seed[overlap] - sample[overlap]
                bias = float(np.nanmean(diff))
                if np.isfinite(bias):
                    sample = sample + bias

            # 2. Data-driven 軟條件：在 overlap 區域往 seed 靠，
            #    但「強度由資料決定」
            if overlap.sum() > 1:
                alpha_i = self._compute_alpha(sample, overlap)
                # alpha_i 越接近 1 → 越信 sample
                # alpha_i 越接近 0 → 越信 seed
                sample[overlap] = (
                    alpha_i * sample[overlap]
                    + (1.0 - alpha_i) * self.seed[overlap]
                )

            # 3. 加權雜訊（與採樣機率成反比）
            valid_mask = np.isfinite(sample)
            raw_weight = float(weights[i])

            # 機率越高，雜訊越小；加上 0.1 避免除零
            noise_scale = self.base_noise / (raw_weight * 100.0 + 0.1)
            noise = self.rng.randn(self.T) * noise_scale

            sample[valid_mask] += noise[valid_mask]
            samples.append(sample)

        return np.stack(samples)


# =============================================================================
# 3. 隨機貝氏聯合補值器 (Bayesian Joint Imputer) - One-Shot
# =============================================================================

class BayesianJointImputer:
    """
    對 raw ensemble 做一次 Multivariate Gaussian 條件抽樣，
    對每一列樣本一次性補齊所有 NaN。
    """

    def __init__(
        self,
        raw_samples: np.ndarray,
        weights: np.ndarray | None = None,
        random_state: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        raw_samples : np.ndarray
            (N, T) 的 raw ensemble，可能含 NaN。
        weights : np.ndarray 或 None
            若 None，視為等權；否則用於加權均值/共變異數。
        """
        self.raw_samples = np.asarray(raw_samples, float)
        self.N, self.T = self.raw_samples.shape
        self.rng = np.random.RandomState(random_state)

        X = self.raw_samples  # (N, T)

        # --------------------------------------------------
        # 1. 估計 mu, Sigma
        # --------------------------------------------------
        if weights is None:
            # 等權：重抽樣次數本身就是隱含權重
            df = pd.DataFrame(X)
            mu = df.mean().values          # (T,)
            Sigma = df.cov(min_periods=2).values  # (T, T)
        else:
            # 顯式 sample weight
            w = np.asarray(weights, float).reshape(-1)
            if w.shape[0] != self.N:
                raise ValueError(
                    f"weights 長度 {w.shape[0]} 與樣本數 {self.N} 不一致"
                )

            w = np.clip(w, 1e-12, None)
            w = w / np.sum(w)

            mu = np.nansum(w[:, None] * X, axis=0)  # (T,)

            X_centered = X - mu[None, :]
            mask = np.isfinite(X_centered)
            X_centered = np.where(mask, X_centered, 0.0)
            Sigma = X_centered.T @ (w[:, None] * X_centered)

        self.mu = mu
        self.Sigma = Sigma

        # --------------------------------------------------
        # 2. 處理 mu 中的 NaN
        # --------------------------------------------------
        mask_nan_mu = np.isnan(self.mu)
        if mask_nan_mu.any():
            x = np.arange(self.T)
            valid = ~mask_nan_mu
            if valid.sum() > 0:
                self.mu[mask_nan_mu] = np.interp(
                    x[mask_nan_mu],
                    x[valid],
                    self.mu[valid],
                )
            else:
                self.mu[:] = 0.0

        # --------------------------------------------------
        # 3. 處理 Sigma 的 NaN / 0 對角線，確保可逆
        # --------------------------------------------------
        if np.isnan(self.Sigma).any():
            fallback_var = np.nanmean(np.diag(self.Sigma))
            if not np.isfinite(fallback_var):
                fallback_var = 1.0

            inds = np.where(np.isnan(self.Sigma))
            self.Sigma[inds] = 0.0

            for i in range(self.T):
                if (self.Sigma[i, i] == 0.0) or (not np.isfinite(self.Sigma[i, i])):
                    self.Sigma[i, i] = fallback_var

        # 對稱化 + 預處理
        self.Sigma = self._make_psd(self.Sigma, min_variance=1e-4)

    def _make_psd(self, matrix: np.ndarray, min_variance: float = 1e-6) -> np.ndarray:
        """
        強制將矩陣轉換為對稱正定矩陣 (Symmetric Positive Definite)。
        透過特徵值分解，將負的特徵值截斷為 min_variance。
        """
        # 1. 強制對稱
        matrix = (matrix + matrix.T) / 2.0
        
        # 2. 特徵值分解 (使用 eigh 針對 Hermitian/Symmetric 矩陣更穩)
        try:
            vals, vecs = np.linalg.eigh(matrix)
            
            # 3. 修正特徵值：所有小於 min_variance 的值都設為 min_variance
            vals = np.maximum(vals, min_variance)
            
            # 4. 重組矩陣
            reconstructed = (vecs * vals) @ vecs.T
            
            # 5. 再次強制對稱 (消除浮點數誤差)
            return (reconstructed + reconstructed.T) / 2.0
        except np.linalg.LinAlgError:
            # 萬一分解失敗，退回對角矩陣
            return np.eye(matrix.shape[0]) * min_variance

    def impute(self) -> np.ndarray:
        """
        執行單次隨機條件採樣補值。
        """
        X = np.copy(self.raw_samples)
        X_filled = np.copy(X)

        for i in range(self.N):
            sample = X[i]
            mask = np.isfinite(sample)
            missing = ~mask

            if missing.sum() == 0:
                continue

            idx_o = np.where(mask)[0]
            idx_u = np.where(missing)[0]

            # 如果完全沒有觀測值，就直接整段從 N(mu, Sigma) 抽
            if len(idx_o) == 0:
                try:
                    X_filled[i] = self.rng.multivariate_normal(
                        self.mu, self.Sigma, check_valid='warn'
                    )
                except:
                    X_filled[i] = self.mu
                continue

            mu_o = self.mu[idx_o]
            mu_u = self.mu[idx_u]

            # 加強穩定性：對角線加上微量噪音
            Sigma_oo = self.Sigma[np.ix_(idx_o, idx_o)]
            Sigma_oo = self._make_psd(Sigma_oo, min_variance=1e-5)
            
            Sigma_uo = self.Sigma[np.ix_(idx_u, idx_o)]
            Sigma_uu = self.Sigma[np.ix_(idx_u, idx_u)]

            try:
                # 使用 solve 求解 gain_T (Sigma_oo^-1 * Sigma_uo^T)
                gain_T = np.linalg.solve(Sigma_oo, Sigma_uo.T)
                
                cond_mu = mu_u + gain_T.T @ (sample[idx_o] - mu_o)
                
                # 計算條件共變異數：Sigma_uu - Sigma_uo * Sigma_oo^-1 * Sigma_uo^T
                cond_sigma = Sigma_uu - gain_T.T @ Sigma_uo.T

                # [關鍵修復]：強制清洗 cond_sigma 確保正定
                cond_sigma = self._make_psd(cond_sigma, min_variance=1e-8)

                # 抽樣
                noise = self.rng.multivariate_normal(
                    np.zeros(len(idx_u)),
                    cond_sigma,
                    check_valid='warn' # 允許微小誤差，因為我們已經手動修復過
                )
                X_filled[i, idx_u] = cond_mu + noise

            except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                # 如果還發生錯誤（極罕見），退回均值填補
                X_filled[i, idx_u] = mu_u

        return X_filled


# =============================================================================
# 4. 總流程 Wrapper (Generator) - 對外接口
# =============================================================================

class Generator:
    """
    整合貝氏時序生成流程的總控制器。
    """

    def __init__(
        self,
        seed: np.ndarray,
        datapreparer,
        window_size: int,
        max_shift: int,
        top_k: int,
        random_state: int = 0,
    ) -> None:
        self.seed = np.asarray(seed, float)
        self.window_size = window_size
        self.max_shift = max_shift
        self.top_k = top_k
        self.random_state = random_state

        # 準備歷史子序列
        grouped = datapreparer.generate_grouped_subsequences()

        all_subseq: list[np.ndarray] = []
        for _, arrs in grouped.items():
            for arr in arrs:
                if len(arr) == window_size:
                    all_subseq.append(np.asarray(arr, float))

        # 限制樣本數，順便估 base_noise
        all_subseq = all_subseq[:1000]
        N = len(all_subseq)

        if N > 0:
            sample_size = min(N, 100)
            rng = np.random.RandomState(random_state)
            idx = rng.choice(N, sample_size, replace=False)
            pool = np.stack([all_subseq[i] for i in idx])

            base_noise = float(np.nanmean(np.nanstd(pool, axis=0)))
            if base_noise < 1e-6:
                base_noise = 1e-3
            self.base_noise = base_noise
        else:
            self.base_noise = 0.1

        self.estimator = BayesianEstimator(
            samples=all_subseq,
            window_size=window_size,
            max_shift=max_shift,
            top_k=top_k,
        )

    def generate(self, n_sample: int = 100) -> Tuple[np.ndarray, dict]:
        """
        執行完整的時序樣本生成流程，並記錄每個步驟的耗時。

        Returns
        -------
        complete_samples : np.ndarray
            shape = (n_sample, T)
        timings : dict
            {"step1_resample": ..., "step2_mixture": ..., "step3_impute": ..., "total": ...}
        """
        timings: dict[str, float] = {}

        # 1. 機率重抽樣
        t0 = time.perf_counter()
        resampled_pool, weights = self.estimator.get_resampled_pool(
            self.seed,
            n_draws=n_sample,
            selection_mode="topk",
        )
        timings["step1_resample"] = time.perf_counter() - t0

        # 2. 混合與加權雜訊（條件在 seed 上）
        t0 = time.perf_counter()
        mixture_gen = BayesianMixtureGenerator(
            seed=self.seed,
            base_noise=self.base_noise,
            random_state=self.random_state,
        )
        raw_samples = mixture_gen.generate_raw_ensemble(
            pool=resampled_pool,
            weights=weights,
        )
        timings["step2_mixture"] = time.perf_counter() - t0

        # 3. 單次隨機聯合補值（Joint MVN）
        t0 = time.perf_counter()
        imputer = BayesianJointImputer(
            raw_samples,
            weights=None,  # 重抽樣次數已經 encode 權重，這裡用等權即可
            random_state=self.random_state,
        )
        complete_samples = imputer.impute()
        timings["step3_impute"] = time.perf_counter() - t0

        timings["total"] = sum(timings.values())
        return complete_samples, timings
