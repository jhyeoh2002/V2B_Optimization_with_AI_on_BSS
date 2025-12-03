## 3.1 Similarity Assessment and Bayesian Resampling

### 3.1.1 Sequence Preprocessing and Temporal Alignment

Prior to similarity calculation, all time series $X$ are standardized using the Z-Score normalization $\mathcal{N}(\cdot)$ to eliminate scale and mean biases.

$$
X^{\text{norm}} = \mathcal{N}(X) = \frac{X - \text{Mean}(X)}{\text{Std}(X)}
$$

For the target sequence $X^{\text{seed}}$ and a historical candidate sequence $s_i$, the optimal time lag $\tau^*$ is determined using the maximum Pearson Cross-Correlation $C(\cdot, \cdot)$ to achieve temporal alignment $\hat{s}_i = \operatorname{Roll}(s_i, -\tau^*)$.

$$
\tau^* = \underset{\tau \in [-\tau_{\max}, \tau_{\max}]}{\operatorname{argmax}} \left( C(\mathcal{N}(X^{\text{seed}}), \mathcal{N}(\operatorname{Roll}(s_i, -\tau))) \right)
$$

### 3.1.2 Similarity Quantification and Probability Weighting $P_i$

The Euclidean Distance $D_i$ on the standardized sequences is employed as the fundamental similarity metric and subsequently converted into a sampling probability $P_i$. The probability $P_i$ is inversely proportional to the square of the distance and is normalized via the Softmax function.

$$
D_i = \operatorname{RMSE}(\mathcal{N}(X^{\text{seed}}), \mathcal{N}(\hat{s}_i))
$$

$$
W_i \propto \frac{1}{D_i^2} \implies P_i = \frac{W_i}{\sum_{j} W_j}
$$

### 3.1.3 Ensemble Resampling

Based on the probability distribution $\mathbf{P} = \{P_1, \ldots, P_N\}$, $n_{\text{draws}}$ samples are drawn with replacement to form the resampled pool $\mathcal{P} = \{\hat{s}_{k}\}_{k=1}^{n_{\text{draws}}}$. Each sample's original sampling probability $P^{\text{draw}}_k$ is recorded.

---

## 3.2 Structure Fusion and Stochasticity Injection

In this stage, samples from $\mathcal{P}$ are fused with $X^{\text{seed}}$, and weighted noise is introduced to form the raw ensemble $X^{\text{raw}}$, which contains missing values.

### 3.2.1 Fusion and Observation Locking

First, the mean bias $\Delta \mu_k$ between $\hat{s}_k$ and $X^{\text{seed}}$ over the overlapping interval is computed and corrected. Subsequently, the known observations from $X^{\text{seed}}$ are overlaid onto the corrected sample, thus locking the known information.

### 3.2.2 Inverse-Probability Weighted Noise Injection

To increase ensemble dispersion, weighted Gaussian noise $\delta_k \sim \mathcal{N}(0, \sigma_{\text{noise}, k}^2 \cdot I)$ is injected into each sample. The noise scale $\sigma_{\text{noise}, k}$ is inversely proportional to the sample's original sampling probability $P^{\text{draw}}_k$, ensuring that samples with lower similarity contribute higher levels of stochasticity.

$$
\sigma_{\text{noise}, k} \propto \frac{1}{P^{\text{draw}}_k}
$$

The raw ensemble $X^{\text{raw}}$ is composed of the fused and perturbed samples.

---

## 3.3 Stochastic Bayesian Joint Imputation

The final stage employs the conditional sampling mechanism of the Multivariate Gaussian Distribution (MVN) to perform a **One-Shot Stochastic Imputation** on all missing values in $X^{\text{raw}}$, preserving the joint covariance structure of the ensemble.

### 3.3.1 MVN Parameter Estimation

The global mean vector $\boldsymbol{\mu}$ and the covariance matrix $\boldsymbol{\Sigma}$ are estimated from the raw ensemble $X^{\text{raw}}$:
$$
\boldsymbol{\mu} = \operatorname{Mean}(X^{\text{raw}}) \quad ; \quad \boldsymbol{\Sigma} = \operatorname{Cov}(X^{\text{raw}})
$$

### 3.3.2 Conditional Sampling Imputation

For any sample $X_k$ in $X^{\text{raw}}$, it is partitioned into observed values $\mathbf{X}_o$ and missing values $\mathbf{X}_u$. According to MVN conditional distribution theory, the distribution of the missing values $\mathbf{X}_u \mid \mathbf{X}_o$ is Gaussian: $\mathcal{N}(\boldsymbol{\mu}_{u \mid o}, \boldsymbol{\Sigma}_{u \mid o})$.

The conditional mean $\boldsymbol{\mu}_{u \mid o}$ and the conditional covariance $\boldsymbol{\Sigma}_{u \mid o}$ are calculated by:

$$
\boldsymbol{\mu}_{u \mid o} = \boldsymbol{\mu}_u + \boldsymbol{\Sigma}_{u o} \boldsymbol{\Sigma}_{o o}^{-1} (\mathbf{X}_o - \boldsymbol{\mu}_o)
$$

$$
\boldsymbol{\Sigma}_{u \mid o} = \boldsymbol{\Sigma}_{u u} - \boldsymbol{\Sigma}_{u o} \boldsymbol{\Sigma}_{o o}^{-1} \boldsymbol{\Sigma}_{o u}
$$

Finally, the imputed values $\mathbf{X}^{\text{impute}}_u$ are obtained via a single random draw from $\mathcal{N}(\boldsymbol{\mu}_{u \mid o}, \boldsymbol{\Sigma}_{u \mid o})$, yielding the complete ensemble sample $X^{\text{complete}}$.