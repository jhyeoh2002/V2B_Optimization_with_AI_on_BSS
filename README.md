# V2B Optimization with AI on BSS - Data Documentation

## Overview
This repository contains data and scripts for optimizing Vehicle-to-Building (V2B) energy systems using AI techniques. The data spans from **January 2023 to September 2024** and includes weather and building energy usage data.

---

## Data Sources

1. Gogoro Station Data

    **Data Types**: Number of fully charged batteries.

    **Source**: xxx
1. Weather Data

    **Data Types**: Radiation Data (MJ/㎡) and Temperature Data (℃)

    **Source**: *中央氣象署 (CWA)
  氣候觀測資料查詢服務 Station Data — 測站編號 466920 臺北*, URL: [CWA Station Data](https://codis.cwa.gov.tw/StationData), Accessed: 19 October 2025。

1. Building Data

    **Data Types**: Electricity Usage (kWh)(度)

    **Source**: *國立台灣大學 推廣中心饋線 用電日報表*, URL: [NTU ePower Platform](https://epower.ga.ntu.edu.tw/fn4/report2.aspx), Accessed: 19 October 2025.

---


Class 

## Time Series Generator

A Bayesian sequence modeling toolkit for generating time series samples based on historical patterns and a given seed sequence.  
This package estimates the posterior distribution using historical analogs and produces realistic synthetic samples via weighted KDE.

### 📦 Features

- Time series subsequence extraction and alignment
- Observation-pattern-based grouping
- Bayesian posterior distribution estimation
- Weighted KDE sampling
- Supports missing values (NaN) and non-stationary seed inputs


### 📊 Output Example
Output is a NumPy array of shape (n_sample, window_size)
Each row is a generated time series sample


## Run example

### Installation

```
bash pip install -r requirements.txt
```

### Method 1. Run with python script with default parameters and record in logfiles
```
nohup 
python3 time_series_generator.py > logs/main_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
### Method 2. Run in batch with diffrerent parameters
```
      bash run_batch.sh
```
---

## Contact
This repository is maintained by **JIAN HERN YEOH (r12521626@ntu.edu.tw)** and **JUN-WEI DING (d13521023@ntu.edu.tw)** from **National Taiwan University, Department of Civil Engineering, Computer-Aided Engineering Division**. For questions, feedback, or collaboration inquiries, feel free to open an issue or contact the maintainers directly.
 