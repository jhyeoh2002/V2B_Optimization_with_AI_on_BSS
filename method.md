## 3.1 Similarity Assessment and Bayesian Resampling

### 3.1.1 Sequence Preprocessing and Temporal Alignment

All time series $X$ are standardized using Z-score normalization:

$$
X^{\text{norm}} = \mathcal{N}(X) = \frac{X - \mu_X}{\sigma_X},
$$

where $\mu_X$ and $\sigma_X$ denote the sample mean and standard deviation of $X$.

Let $X^{\text{seed}}$ be the target sequence and $s_i$ be the $i$-th historical candidate.
For each candidate, we consider temporal shifts $\tau \in [-\tau_{\max}, \tau_{\max}]$ and define the rolled sequence

$$
s_i^{(\tau)} = \mathrm{Roll}(s_i, \tau).
$$

We then normalize

$$
X^{\text{seed,norm}} = \mathcal{N}(X^{\text{seed}}), \qquad
s_i^{(\tau),\text{norm}} = \mathcal{N}\!\bigl(s_i^{(\tau)}\bigr).
$$

Let $\Omega_i(\tau)$ be the set of indices where both sequences are finite:

$$
\Omega_i(\tau) = \{\, t \mid X^{\text{seed,norm}}_t \text{ and } s_i^{(\tau),\text{norm}}{}_t \text{ are finite} \,\}.
$$

The distance between $X^{\text{seed}}$ and $s_i$ at lag $\tau$ is

$$
d_i(\tau)
= \left(
\sum_{t \in \Omega_i(\tau)}
\left(
X^{\text{seed,norm}}_t - s_i^{(\tau),\text{norm}}{}_t
\right)^2
\right)^{1/2}.
$$

The optimal lag is chosen to minimize this distance:

$$
\tau^* = \arg\min_{\tau \in [-\tau_{\max}, \tau_{\max}]} d_i(\tau),
$$

and the aligned candidate is

$$
\hat{s}_i = s_i^{(\tau^*)} = \mathrm{Roll}(s_i, \tau^*).
$$

---

### 3.1.2 Similarity Quantification and Probability Weighting

Similarity between $X^{\text{seed}}$ and each aligned candidate $\hat{s}_i$
is quantified by the Euclidean distance on normalized sequences.

Let

$$
\hat{s}_i^{\text{norm}} = \mathcal{N}(\hat{s}_i), \qquad
\Omega_i = \{\, t \mid X^{\text{seed,norm}}_t \text{ and } \hat{s}_i^{\text{norm}}{}_t \text{ are finite} \,\}.
$$

Then

$$
D_i
= \left(
\sum_{t \in \Omega_i}
\left(
X^{\text{seed,norm}}_t - \hat{s}_i^{\text{norm}}{}_t
\right)^2
\right)^{1/2}.
$$

To focus on the most relevant candidates, we retain only the Top-$K$ smallest distances:

$$
\mathcal{I}_{\text{top-}K}
= \left\{
i \,\middle|\, D_i \text{ is among the } K \text{ smallest values}
\right\}.
$$

For $i \in \mathcal{I}_{\text{top-}K}$ we assign power-law weights

$$
W_i = \frac{1}{D_i^{\beta} + \varepsilon},
\qquad
\beta \gg 1 \; (\text{e.g. } \beta = 10), \quad \varepsilon > 0,
$$

and set $W_i = 0$ for all $i \notin \mathcal{I}_{\text{top-}K}$.
The normalized sampling probabilities are

$$
P_i = \frac{W_i}{\sum_{j=1}^N W_j}.
$$

---

### 3.1.3 Ensemble Resampling

Given the probability distribution $\{P_1, \dots, P_N\}$,
we draw $n_{\text{draws}}$ indices $\{ i_k \}_{k=1}^{n_{\text{draws}}}$ **with replacement**:

$$
\mathbb{P}(i_k = i) = P_i, \qquad i = 1, \dots, N.
$$

The resampled, aligned pool is

$$
\mathcal{P}
= \{\hat{s}_k\}_{k=1}^{n_{\text{draws}}},
\qquad
\hat{s}_k = \hat{s}_{i_k}.
$$

Each resampled element keeps its original sampling probability

$$
P_k^{\text{draw}} = P_{i_k}.
$$

---

## 3.2 Structure Fusion and Stochasticity Injection

### 3.2.1 Level Alignment and Data-Driven Soft Conditioning

For each resampled sequence $\hat{s}_k$, we first identify the overlapping indices

$$
\mathcal{O}_k
= \{\, t \mid X^{\text{seed}}_t \text{ and } \hat{s}_{k,t} \text{ are both observed} \,\}.
$$

Over this overlap, we compute the mean bias

$$
\Delta\mu_k
= \frac{1}{|\mathcal{O}_k|}
\sum_{t \in \mathcal{O}_k}
\bigl( X^{\text{seed}}_t - \hat{s}_{k,t} \bigr).
$$

We then apply level alignment

$$
\tilde{s}_{k,t} = \hat{s}_{k,t} + \Delta\mu_k.
$$

Next, we define the seed variance

$$
\sigma_{\text{seed}}^2
= \operatorname{Var}\bigl( X^{\text{seed}}_t \bigr)
$$

over all observed $t$, and the variance of the difference over the overlap

$$
\sigma_{\text{diff},k}^2
= \operatorname{Var}\bigl( \tilde{s}_{k,t} - X^{\text{seed}}_t \bigr)_{t \in \mathcal{O}_k}.
$$

An adaptive blending coefficient is

$$
\alpha_k
= \frac{\sigma_{\text{seed}}^2}{\sigma_{\text{seed}}^2 + \sigma_{\text{diff},k}^2},
\qquad 0 \le \alpha_k \le 1.
$$

The fused sample $X^{\text{raw}}_k$ is defined component-wise as

$$
X^{\text{raw}}_{k,t} =
\begin{cases}
\alpha_k \,\tilde{s}_{k,t} + (1 - \alpha_k)\,X^{\text{seed}}_t, & t \in \mathcal{O}_k, \\[4pt]
\tilde{s}_{k,t}, & t \notin \mathcal{O}_k.
\end{cases}
$$

This soft conditioning shrinks the candidate towards the seed when their discrepancy is large (small $\alpha_k$), and preserves the candidate’s structure when they are already similar (large $\alpha_k$).

---

### 3.2.2 Inverse-Probability Weighted Noise Injection

To enrich ensemble diversity, we inject zero-mean Gaussian noise into each fused sample:

$$
\delta_k \sim \mathcal{N}\bigl(0, \sigma_{\text{noise},k}^2 I\bigr),
$$

with noise scale

$$
\sigma_{\text{noise},k}
= \frac{c_0}{c_1 P_k^{\text{draw}} + c_2},
\qquad c_0, c_1, c_2 > 0.
$$

Thus, lower-similarity samples (smaller $P_k^{\text{draw}}$) receive stronger perturbations,
while highly similar samples remain closer to the seed.

The resulting raw ensemble is

$$
X^{\text{raw}}
= \{ X^{\text{raw}}_k \}_{k=1}^{n_{\text{draws}}}.
$$

---

## 3.3 Stochastic Bayesian Joint Imputation

### 3.3.1 MVN Parameter Estimation

We treat $X^{\text{raw}}$ as $N$ realizations of a $T$-dimensional random vector.
The global mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ are estimated as

$$
\boldsymbol{\mu} = \mathbb{E}[X^{\text{raw}}], \qquad
\boldsymbol{\Sigma} = \operatorname{Cov}(X^{\text{raw}}).
$$

In practice, $\boldsymbol{\Sigma}$ is regularized and projected to be symmetric positive semi-definite for numerical stability.

---

### 3.3.2 Multivariate Gaussian Conditional Sampling

For any ensemble member $X_k$ (a row of $X^{\text{raw}}$),
let $o$ denote the indices of observed entries and $u$ the indices of missing entries.
We partition the mean and covariance as

$$
\boldsymbol{\mu} =
\begin{pmatrix}
\boldsymbol{\mu}_o \\
\boldsymbol{\mu}_u
\end{pmatrix},
\qquad
\boldsymbol{\Sigma} =
\begin{pmatrix}
\boldsymbol{\Sigma}_{oo} & \boldsymbol{\Sigma}_{ou} \\
\boldsymbol{\Sigma}_{uo} & \boldsymbol{\Sigma}_{uu}
\end{pmatrix}.
$$

Under the multivariate normal (MVN) model, the conditional distribution of the missing part $X_u$ given the observed part $X_o$ is

$$
X_u \mid X_o \sim
\mathcal{N}\bigl(\boldsymbol{\mu}_{u\mid o}, \boldsymbol{\Sigma}_{u\mid o}\bigr),
$$

where

$$
\boldsymbol{\mu}_{u\mid o}
= \boldsymbol{\mu}_u
+ \boldsymbol{\Sigma}_{u o}\,\boldsymbol{\Sigma}_{o o}^{-1}\,(X_o - \boldsymbol{\mu}_o),
$$

$$
\boldsymbol{\Sigma}_{u\mid o}
= \boldsymbol{\Sigma}_{u u}
- \boldsymbol{\Sigma}_{u o}\,\boldsymbol{\Sigma}_{o o}^{-1}\,\boldsymbol{\Sigma}_{o u}.
$$

A single random draw

$$
X_u^{\text{impute}} \sim
\mathcal{N}\bigl(\boldsymbol{\mu}_{u\mid o}, \boldsymbol{\Sigma}_{u\mid o}\bigr)
$$

is used to fill the missing values, yielding the completed sequence

$$
X_k^{\text{complete}} = (X_o, X_u^{\text{impute}}).
$$

Applying this procedure to all ensemble members produces a set of fully imputed sequences $\{ X_k^{\text{complete}} \}_{k=1}^N$ that preserve the learned joint covariance structure while incorporating stochastic variability consistent with the MVN model.
