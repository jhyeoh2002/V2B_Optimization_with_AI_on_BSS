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

## 📁 Project Structure
<pre lang="markdown"> ## 📁 Project Structure ``` ├── time_series_generator/ # Package core │ ├── core.py # Generator + distribution estimator │ ├── preprocessing.py # Subsequence processing logic │ ├── density.py # Posterior update (KDE) │ ├── metrics.py # DTW-based distance functions │ ├── utils.py # Helper functions (e.g. safe nansum) │ ├── config.py # Global configuration │ ├── scripts/ # Run scripts and demos │ ├── run_experiment.py # Entry point for CLI-style execution │ └── test.ipynb # Jupyter notebook demo │ ├── Raw_Data/ # Example raw input data │ └── Gogoro/ # Real-world station time series data │ ├── requirements.txt └── README.md ``` </pre>

## 👨‍💻 Development and Maintenance

This package was developed and is actively maintained by **JUN-WEI DING (d13521023@ntu.edu.tw)**. For questions, feedback, or collaboration inquiries, feel free to open an issue or contact the maintainer.
