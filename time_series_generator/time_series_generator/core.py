import numpy as np
import pandas as pd
from typing import List, Tuple, Union

# =============================================================================
# 0. 基礎工具 (Base Utilities)
# =============================================================================

def _norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    計算 Z-Score 標準化 (Standardization)。
    
    將序列轉換為零均值、單位標準差。
    
    Args:
        x: 待標準化的 NumPy 數組。
        eps: 用於數值穩定的極小值，防止標準差為零。
        
    Returns:
        標準化後的數組。
    """
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    
    # 數值穩定性檢查
    if not np.isfinite(s) or s < eps: 
        s = eps
        
    return (x - m) / s

def _best_lag_xcorr(seed: np.ndarray, cand: np.ndarray, max_shift: int) -> np.ndarray:
    """
    尋找最佳的延遲 (Lag) 以對齊 Seed 和 Candidate，並返回對齊後的 Candidate。
    基於最大 Pearson 互相關係數 (Cross-Correlation) 原則。
    
    Args:
        seed: 種子序列 (參考序列)。
        cand: 候選序列。
        max_shift: 允許的最大時間位移量 (正負方向)。
        
    Returns:
        已對齊 (Roll) 的候選序列。
    """
    x = _norm(seed)
    y = _norm(cand)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    
    best_corr, best_lag = -np.inf, 0
    std_eps = 1e-8 

    for lag in range(-max_shift, max_shift + 1):
        # 根據 lag 調整序列切片
        if lag >= 0: 
            xs, ys = x[lag:], y[:n - lag]
        else: 
            xs, ys = x[:n + lag], y[-lag:]
        
        # 處理 NaN 值，只計算有限數值的相關性
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() < 2: 
            continue

        a = xs[mask]
        b = ys[mask]
        
        # 避免標準差為零導致相關係數無意義 (將其視為最低相關)
        if np.std(a) < std_eps or np.std(b) < std_eps:
            corr = -1.0
        else:
            # 計算 Pearson Correlation
            corr = np.corrcoef(a, b)[0, 1]
            if np.isnan(corr): 
                corr = -1.0 

        if corr > best_corr:
            best_corr = corr
            best_lag = lag
            
    # 使用最佳 lag 反向滾動 (np.roll) 候選序列，使其與 seed 對齊
    return np.roll(np.asarray(cand, float), -best_lag)

def _euclidean_distance(seed_norm: np.ndarray, cand_norm: np.ndarray) -> float:
    """
    計算歐式距離 (作為相似度的物理基礎)。
    即標準化序列的均方根差 (Root Mean Square Error, RMSE)。
    
    Args:
        seed_norm: 標準化的種子序列。
        cand_norm: 標準化的候選序列。
        
    Returns:
        歐式距離值。
    """
    mask = np.isfinite(seed_norm) & np.isfinite(cand_norm)
    if mask.sum() == 0: 
        return np.inf
        
    diff = seed_norm[mask] - cand_norm[mask]
    return np.sqrt(np.mean(diff**2))


# =============================================================================
# 1. 貝氏估計器 (Bayesian Estimator) - 機率重抽樣核心
# =============================================================================

class BayesianEstimator:
    """
    用於計算歷史片段 (samples) 相對於給定 seed 的相似度機率，並進行重抽樣。
    """
    def __init__(self, samples: List[np.ndarray], window_size: int, max_shift: int, top_k: int):
        self.samples = samples
        self.window_size = window_size
        self.max_shift = max_shift
        self.top_k = top_k 

    def get_resampled_pool(self, seed: np.ndarray, n_draws: int = 100, 
                           selection_mode: str = 'topk', top_k_limit: Union[int, None] = None) \
                           -> Tuple[np.ndarray, np.ndarray]:
        """
        核心方法：計算所有歷史片段的機率，並重抽樣出 n_draws 條樣本。
        
        Args:
            seed: 用於匹配的時序片段。
            n_draws: 要抽樣出的樣本數量。
            selection_mode: 'softmax' (機率抽樣) 或 'topk' (硬性選取最佳 K 個)。
            top_k_limit: 在 'topk' 模式下，實際要選取的數量 (None 時使用 self.top_k)。
            
        Returns:
            (resampled_pool, resampled_weights): 重抽樣的對齊後樣本池和它們的原始採樣機率。
        """
        seed = np.asarray(seed, float)
        seed_norm = _norm(seed)
        N = len(self.samples)
        
        aligned_candidates = []
        distances = []
        
        # 1. 對齊並計算所有歷史資料的距離
        for i in range(N):
            cand = self.samples[i]
            # 對齊
            aligned_cand = _best_lag_xcorr(seed, cand, self.max_shift)
            # 計算歐式距離
            dist = _euclidean_distance(seed_norm, _norm(aligned_cand))
            
            aligned_candidates.append(aligned_cand)
            distances.append(dist)
            
        distances = np.array(distances)
        
        # 2. 根據模式計算機率 (Probabilities Calculation)
        
        if selection_mode == 'topk':
            # A. Top-K 模式：硬性選取最佳 K 個，並給予均等機率
            K = min(top_k_limit if top_k_limit is not None else self.top_k, N)
            best_indices = np.argsort(distances)[:K]
            
            probs = np.zeros(N)
            if K > 0:
                probs[best_indices] = 1.0 / K 
                probs /= np.sum(probs) # 確保總和為 1

        else: # Softmax (default)
            # B. Softmax 模式：機率與距離平方成反比 (距離越近，機率越高)
            epsilon = 1e-6
            # 權重 = 1 / (距離^2 + epsilon)
            weights = 1.0 / (distances**2 + epsilon) 
            probs = weights / np.sum(weights)
        
        # 3. 重抽樣 (Resampling)
        
        # 根據機率分佈 (p=probs) 進行 n_draws 次帶放回的抽樣
        indices = np.random.choice(N, size=n_draws, p=probs, replace=True)
        
        resampled_pool = np.stack([aligned_candidates[i] for i in indices])
        resampled_weights = probs[indices] # 記錄它們被選中的原始機率
        
        return resampled_pool, resampled_weights

# =============================================================================
# 2. 混合生成器 (Mixture Generator) - 結合與加噪
# =============================================================================
class BayesianMixtureGenerator:
    """
    將重抽樣的歷史片段與 seed 混合，並加入加權觀測雜訊。
    """
    def __init__(self, seed: np.ndarray, base_noise: float = 0.1, random_state: int = 0):
        self.seed = np.asarray(seed, float)
        self.T = len(seed)
        self.base_noise = base_noise
        self.rng = np.random.RandomState(random_state)

    def generate_raw_ensemble(self, pool: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        直接使用 Resampled Pool 進行混合，產生 Raw Ensemble。
        
        Args:
            pool: 對齊後的重抽樣樣本池 (N x T)。
            weights: 每個樣本在原始分佈中被選中的機率。
            
        Returns:
            混合後的原始樣本集 (Raw Ensemble)。
        """
        n_sample = len(pool)
        samples = []
        seed_mask = np.isfinite(self.seed)
        
        for i in range(n_sample):
            sample = np.copy(pool[i])
            
            # 1. 簡單對齊 (Shift Correction) - 消除均值差異
            cand_mask = np.isfinite(sample)
            overlap = seed_mask & cand_mask
            if overlap.sum() > 0:
                # 計算重疊區域的偏差，並將樣本平移
                diff = self.seed[overlap] - sample[overlap]
                bias = np.mean(diff)
                sample = sample + bias
            
            # 2. 種子覆蓋 - 確保已知點精確保留
            sample[seed_mask] = self.seed[seed_mask]
            
            # 3. 加權雜訊 (Stochasticity Injection)
            valid_mask = np.isfinite(sample)
            raw_weight = weights[i]
            
            # 雜訊規模：與採樣機率成反比 (機率越高，雜訊越小，保留樣本特徵越多)
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
    使用 One-Shot 隨機補值。
    利用多元高斯分佈 (MVN) 的條件機率，對 Raw Samples 中的缺失值進行單次隨機採樣補值，
    以保留最大變異數結構。
    """
    def __init__(self, raw_samples: np.ndarray, random_state: int = 0):
        self.raw_samples = np.asarray(raw_samples)
        self.N, self.T = self.raw_samples.shape
        self.rng = np.random.RandomState(random_state)
        
        # --- 1. 初始化全域參數 (Mu & Sigma) ---
        df = pd.DataFrame(self.raw_samples)
        self.mu = df.mean().values # 全域均值
        
        # 處理 mu 中的 NaN (使用線性插值或設為零)
        mask_nan_mu = np.isnan(self.mu)
        if mask_nan_mu.any():
            x = np.arange(self.T); valid = ~mask_nan_mu
            if valid.sum() > 0:
                self.mu[mask_nan_mu] = np.interp(x[mask_nan_mu], x[valid], self.mu[valid])
            else: 
                self.mu[:] = 0.0
                
        # 計算初始 Covariance (Sigma) - 捕捉樣本池的聯合變異結構
        self.Sigma = df.cov(min_periods=2).values
        
        # 數值穩定和填補
        if np.isnan(self.Sigma).any():
            fallback_var = np.nanmean(np.diag(self.Sigma))
            if np.isnan(fallback_var): fallback_var = 1.0
            inds = np.where(np.isnan(self.Sigma))
            self.Sigma[inds] = 0
            for i in range(self.T):
                if self.Sigma[i, i] == 0: 
                    self.Sigma[i, i] = fallback_var
                    
        # 加上極小的抖動 (Jitter) 確保矩陣可逆
        self.Sigma += np.eye(self.T) * 1e-4

    def impute(self) -> np.ndarray:
        """
        執行單次隨機條件採樣補值。
        
        Returns:
            已完成補值的樣本集 (Complete Samples)。
        """
        X = np.copy(self.raw_samples)
        X_filled = np.copy(X)
        
        # 移除迭代，只執行一次 E-Step (Conditional Sampling)
        for i in range(self.N):
            sample = X[i]
            mask = np.isfinite(sample)
            missing = ~mask
            
            if missing.sum() == 0: 
                continue # 完整樣本跳過
            
            idx_o = np.where(mask)[0]    # 已觀測 (Observed) 索引
            idx_u = np.where(missing)[0] # 未觀測 (Unobserved) 索引
            
            # 處理全空樣本 (罕見): 隨機採樣一條線
            if len(idx_o) == 0:
                X_filled[i] = self.rng.multivariate_normal(self.mu, self.Sigma)
                continue

            # 提取均值和協方差矩陣區塊
            mu_o = self.mu[idx_o]; mu_u = self.mu[idx_u]
            Sigma_oo = self.Sigma[np.ix_(idx_o, idx_o)] + np.eye(len(idx_o)) * 1e-5 # 確保可逆性
            Sigma_uo = self.Sigma[np.ix_(idx_u, idx_o)]
            Sigma_uu = self.Sigma[np.ix_(idx_u, idx_u)]
            
            try:
                # 1. 計算條件平均 (Conditional Mean)
                # Cond_Mu = Mu_u + Sigma_uo * Sigma_oo^-1 * (X_o - Mu_o)
                gain_T = np.linalg.solve(Sigma_oo, Sigma_uo.T) 
                cond_mu = mu_u + gain_T.T @ (sample[idx_o] - mu_o)
                
                # 2. 計算條件變異數 (Conditional Variance)
                # Cond_Sigma = Sigma_uu - Sigma_uo * Sigma_oo^-1 * Sigma_ou
                cond_sigma = Sigma_uu - gain_T.T @ Sigma_uo.T
                
                # 3. 採樣 (Stochastic Sampling)
                # 確保條件協方差矩陣為對稱、正定 (加上 Jitter)
                cond_sigma = (cond_sigma + cond_sigma.T) / 2 + np.eye(len(idx_u)) * 1e-6
                noise = self.rng.multivariate_normal(np.zeros(len(idx_u)), cond_sigma)
                
                X_filled[i, idx_u] = cond_mu + noise # 補值 = 條件平均 + 隨機性
                
            except np.linalg.LinAlgError:
                # 數值失敗時，退化為填補條件平均值 (或全域均值)
                X_filled[i, idx_u] = mu_u

        return X_filled


# =============================================================================
# 4. 總流程 Wrapper (Generator) - 對外接口
# =============================================================================

class Generator:
    """
    整合貝氏時序生成流程的總控制器。
    """
    def __init__(self, seed: np.ndarray, datapreparer, window_size: int, max_shift: int, 
                 top_k: int, random_state: int = 0):
        
        self.seed = np.asarray(seed, float)
        self.window_size = window_size
        self.max_shift = max_shift
        self.top_k = top_k
        self.random_state = random_state

        # 準備歷史資料 (Dataparparer 假定是外部提供的資料準備類)
        grouped = datapreparer.generate_grouped_subsequences()
        all_subseq = []
        for key, arrs in grouped.items():
            for arr in arrs:
                if len(arr) == window_size:
                    all_subseq.append(np.asarray(arr, float))
        
        # 自動計算 base_noise：根據歷史樣本的平均標準差設定雜訊基礎，以保持數據尺度一致
        N = len(all_subseq)
        if N > 0:
            sample_size = min(N, 100)
            rng = np.random.RandomState(random_state)
            idx = rng.choice(N, sample_size, replace=False)
            pool = np.stack([all_subseq[i] for i in idx])
            # 計算所有時間點的平均標準差
            self.base_noise = np.nanmean(np.nanstd(pool, axis=0)) 
            if self.base_noise < 1e-6: 
                self.base_noise = 1e-3
        else: 
            self.base_noise = 0.1

        self.estimator = BayesianEstimator(all_subseq, window_size, max_shift, top_k)

    def generate(self, n_sample: int = 100) -> np.ndarray:
        """
        執行完整的時序樣本生成流程。
        
        Args:
            n_sample: 要生成的時序樣本數量。
            
        Returns:
            (n_sample, T) 維度的完整時序樣本矩陣。
        """
        # 1. 機率重抽樣
        resampled_pool, weights = self.estimator.get_resampled_pool(
            self.seed, n_draws=n_sample
        )

        # 2. 混合與加權雜訊
        mixture_gen = BayesianMixtureGenerator(
            seed=self.seed, base_noise=self.base_noise, random_state=self.random_state
        )
        raw_samples = mixture_gen.generate_raw_ensemble(pool=resampled_pool, weights=weights)

        # 3. 單次隨機聯合補值
        imputer = BayesianJointImputer(raw_samples, random_state=self.random_state)
        complete_samples = imputer.impute()

        return complete_samples