# High-Performance Time Series Transformation with NumPy & pandas

## 📌 Overview
This project implements efficient time series transformations using NumPy and pandas and compares their performance. The focus is on handling large datasets (≥ 1M rows) with optimized routines.

## 🚀 Features
- Rolling window statistics (mean, variance)  
- Exponentially Weighted Moving Averages (EWMA)  
- FFT-based spectral analysis  
- Band-pass filtering  
- Benchmarking of NumPy (stride tricks) vs. pandas (built-ins)  
- CSV + chart outputs for runtime comparisons  

## 📂 File Structure
```
├── timeseries_utils.py       # Core implementations (NumPy + pandas)
├── benchmark.py              # Benchmarking script
├── benchmark_results.csv     # Results saved after running benchmarks
├── benchmark_plot.png        # Performance chart
├── report.md                 # Summary of performance trade-offs
└── README.md                 # Project documentation
```

## ⚙️ Installation
### Requirements
- Python 3.11+ (tested on Python 3.13 without Numba)
- Libraries:
```
numpy
pandas
matplotlib
```
### Install dependencies
```
pip install -r requirements.txt
```

## ▶️ Usage

1. Run benchmarks:
```
python benchmark.py
```
This will generate:
- benchmark_results.csv → runtime comparisons
- benchmark_plot.png → visualization

2. Import functions in your own projects:
```
import timeseries_utils as ts
import numpy as np

data = np.random.randn(1000, 3)
means, vars_ = ts.rolling_mean_var_numpy(data, window=50)
```

## 📖 Report

Detailed methodology, benchmarks, and recommendations are available in report.md

## 🔮 Future Work

- Add GPU acceleration (CuPy, RAPIDS)
- Integrate Numba (when stable for Python 3.13)
- Adaptive method selector based on dataset size