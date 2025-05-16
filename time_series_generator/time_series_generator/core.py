import numpy as np
from collections import defaultdict
from sklearn.neighbors import KernelDensity
from time_series_generator.preprocessing  import DataPrepare
from time_series_generator.metrics import fast_dtw_distance
from time_series_generator.density import compute_posterior_weights_from_partial_subseq
import time_series_generator.config as cfg

class Generator:
    def __init__(self,
                 window_size=cfg.WINDOW_SIZE,
                 resolution=cfg.RESOLUTION,
                 seed=cfg.SEED,
                 n_sample=cfg.NSAMPLE,
                 bandwidth=cfg.BANDWIDTH,
                 random_state=cfg.RANDOM_STATE):
        self.window_size = window_size
        self.resolution = resolution
        self.seed = seed
        self.n_sample = n_sample
        self.bandwidth = bandwidth
        self.random_state = random_state

        self._estimator = BayesianDistributionEstimator(
            window_size=self.window_size,
            resolution=self.resolution
        )

    def generate(self):
        mean_seed, std_seed, X_joint, w_post = self._estimator.estimate_and_correct_distribution(seed=self.seed)

        kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        kde.fit(X_joint, sample_weight=w_post)

        new_samples = kde.sample(n_samples=self.n_sample, random_state=self.random_state)
        new_samples = new_samples * std_seed + mean_seed

        return new_samples
    
class BayesianDistributionEstimator:
    """進行時間序列 seed 的貝式分布估計與修正"""

    def __init__(self, window_size=cfg.WINDOW_SIZE, resolution=cfg.RESOLUTION):
        self.window_size = window_size
        self.resolution = resolution
        self.datapreparer = DataPrepare(window_size, resolution)
        self.grouped_samples = None

    def estimate_and_correct_distribution(self, seed: np.ndarray, bandwidth=cfg.BANDWIDTH):
        if self.grouped_samples is None:
            self._prepare_data()

        mean_seed, std_seed, normal_seed = self._normalize_seed(seed)

        keys = list(self.grouped_samples.keys())
        if not keys:
            raise ValueError("No grouped samples available.")

        # Step 1: 初始化先驗分布
        key = keys[0]
        history = np.array(self.grouped_samples[key])
        distances = fast_dtw_distance(normal_seed, history)
        weights = 1 / (distances**2 + 1e-8)
        weights /= np.sum(weights)

        p_weighted = defaultdict(float)
        for row, weight in zip(history, weights):
            k = tuple(row[~np.isnan(row)])
            p_weighted[k] += weight
        total = sum(p_weighted.values())
        for k in p_weighted:
            p_weighted[k] /= total

        X_joint = history
        w_prior = np.array(list(p_weighted.values()))
        w_post = w_prior

        # Step 2: 根據其他 pattern 修正後驗分布
        for i in range(1, len(keys)):
            key = keys[i]
            history = np.array(self.grouped_samples[key])
            if history.shape[0] < 30:
                continue

            distances = fast_dtw_distance(normal_seed, history)
            weights = 1 / (distances**2 + 1e-8)
            weights /= np.sum(weights)

            X_marginal_obs = history
            w_obs = weights

            w_post = compute_posterior_weights_from_partial_subseq(
                X_joint, w_post,
                X_marginal_obs, w_obs,
                bandwidth=bandwidth
            )

        return mean_seed, std_seed, X_joint, w_post

    def _prepare_data(self):
        self.grouped_samples = self.datapreparer.generate_grouped_subsequences()

    def _normalize_seed(self, seed):
        mean_val = np.mean(seed)
        std_val = np.std(seed)

        if np.allclose(seed, seed[0]):
            normal = seed / mean_val if mean_val != 0 else np.zeros_like(seed)
        else:
            normal = (seed - mean_val) / std_val if std_val != 0 else np.zeros_like(seed)

        return mean_val, std_val, normal
