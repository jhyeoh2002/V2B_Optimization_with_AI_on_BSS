import numpy as np
from scipy.spatial.distance import cdist

def kde_logpdf_weighted(X_query, X_sample, weights, bandwidth):
    """
    使用加權 KDE（Gaussian）估計 log pdf
    """
    d2 = cdist(X_query, X_sample, metric='sqeuclidean')  # shape (Q, N)
    kernel_vals = np.exp(-d2 / (2 * bandwidth**2))
    weighted_kernels = kernel_vals * weights[None, :]  # broadcasting
    density = np.sum(weighted_kernels, axis=1) + 1e-12  # 避免除 0
    norm = (1 / (np.sqrt(2 * np.pi) * bandwidth))**X_sample.shape[1]
    return np.log(norm * density)

def compute_posterior_weights_from_partial_subseq(
    X_joint,               # (N, t): joint samples
    w_prior,               # (N, ): prior weights
    X_marginal_obs,        # (M, m): observed marginal with NaNs allowed
    w_obs,                 # (M, ): obs weights
    # observed_dims,         # (k, ): indices of observed dimensions (e.g., [0,2])
    bandwidth=0.5
):
    # 驗證
    N, t = X_joint.shape
    M, m = X_marginal_obs.shape
    observed_dims = np.where(~np.isnan(X_marginal_obs[0]))[0]
    assert len(w_prior) == N, "w_prior 長度錯誤"
    assert len(w_obs) == M, "w_obs 長度錯誤"
 
    # Normalize weights
    w_prior = w_prior / np.sum(w_prior)
    w_obs = w_obs / np.sum(w_obs)

    # 萃取有效維度
    X_joint_sub = X_joint[:, observed_dims]                # shape (N, k)
    X_obs_sub = X_marginal_obs[:, observed_dims]           # shape (M, k)

    # KDE
    log_p_obs = kde_logpdf_weighted(X_joint_sub, X_obs_sub, w_obs, bandwidth)
    log_p_prior = kde_logpdf_weighted(X_joint_sub, X_joint_sub, w_prior, bandwidth)

    # 計算後驗權重
    log_w_post = log_p_obs - log_p_prior + np.log(w_prior + 1e-10)
    w_post = np.exp(log_w_post)
    w_post /= np.sum(w_post)

    return w_post