import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeseries_utils as ts

# Benchmark Config
N_ROWS = 1_000_000  
N_FEATURES = 3
WINDOW = 50
ALPHA = 0.1

# Generate synthetic dataset
np.random.seed(42)
data = np.random.randn(N_ROWS, N_FEATURES)
df = pd.DataFrame(data, columns=[f"f{i}" for i in range(N_FEATURES)])

# Helper to time functions
def benchmark(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

# Rolling Mean/Var
print("Benchmarking Rolling Mean/Var...")

_, t_numpy = benchmark(ts.rolling_mean_var_numpy, data, WINDOW)
_, t_pandas = benchmark(ts.rolling_mean_var_pandas, df, WINDOW)

print(f"NumPy: {t_numpy:.3f}s | pandas: {t_pandas:.3f}s")

# EWMA
print("\nBenchmarking EWMA...")

_, t_numpy = benchmark(ts.ewma_numpy, data[:, 0], ALPHA)
_, t_pandas = benchmark(ts.ewma_pandas, df[['f0']], ALPHA)

print(f"NumPy: {t_numpy:.3f}s | pandas: {t_pandas:.3f}s")

# FFT Spectrum
print("\nBenchmarking FFT Spectrum...")

_, t_fft = benchmark(ts.fft_spectrum, data[:, 0])
print(f"FFT Spectrum: {t_fft:.3f}s")

# Band-pass Filter
print("\nBenchmarking Band-pass Filter...")

_, t_filter = benchmark(ts.bandpass_filter, data[:, 0], 0.1, 0.5)
print(f"Band-pass Filter: {t_filter:.3f}s")

# Collect results
results = {
    "Task": ["Rolling Mean/Var", "EWMA", "FFT Spectrum", "Band-pass Filter"],
    "NumPy (s)": [t_numpy, t_numpy, t_fft, t_filter],
    "pandas (s)": [t_pandas, t_pandas, None, None]
}

results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv("Task 3/benchmark_results.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(results_df["Task"], results_df["NumPy (s)"], label="NumPy")
if results_df["pandas (s)"].notnull().any():
    ax.bar(results_df["Task"], results_df["pandas (s)"].fillna(0),
           alpha=0.7, label="pandas")

ax.set_ylabel("Runtime (s)")
ax.set_title("NumPy vs pandas Performance")
ax.legend()
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("Presentation/Task3_benchmark_plot.png")
plt.show()
