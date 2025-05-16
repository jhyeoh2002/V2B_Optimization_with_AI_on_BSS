# Time Series Generator

A Bayesian sequence modeling toolkit for generating time series samples based on historical patterns and a given seed sequence.  
This package estimates the posterior distribution using historical analogs and produces realistic synthetic samples via weighted KDE.

## 📦 Features

- Time series subsequence extraction and alignment
- Observation-pattern-based grouping
- Bayesian posterior distribution estimation
- Weighted KDE sampling
- Supports missing values (NaN) and non-stationary seed inputs


## 📊 Output Example
Output is a NumPy array of shape (n_sample, window_size)
Each row is a generated time series sample

## 👨‍💻 Development and Maintenance

This package was developed and is actively maintained by **JUN-WEI DING (d13521023@ntu.edu.tw)**. For questions, feedback, or collaboration inquiries, feel free to open an issue or contact the maintainer.
