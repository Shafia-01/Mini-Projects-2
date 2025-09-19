import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Rolling Window Statistics
def rolling_mean_var_numpy(arr, window: int):
    if arr.ndim == 1:
        arr = arr[:, None]

    windows = sliding_window_view(arr, (window, arr.shape[1]))[:, 0, :, :]
    means = windows.mean(axis=1)
    vars_ = windows.var(axis=1)
    return means, vars_

def rolling_mean_var_pandas(df: pd.DataFrame, window: int):
    return df.rolling(window=window).mean(), df.rolling(window=window).var()

# Exponentially Weighted Moving Averages (EWMA)
def ewma_numpy(arr, alpha: float):
    arr = np.asarray(arr, dtype=float)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for t in range(1, len(arr)):
        out[t] = alpha * arr[t] + (1 - alpha) * out[t-1]
    return out

def ewma_pandas(df: pd.DataFrame, alpha: float):
    return df.ewm(alpha=alpha).mean()

# FFT-based Spectral Analysis
def fft_spectrum(arr, sampling_rate=1.0):
    n = len(arr)
    freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
    fft_vals = np.fft.rfft(arr)
    power = np.abs(fft_vals) ** 2
    return freqs, power

# Band-pass Filtering
def bandpass_filter(arr, low_freq, high_freq, sampling_rate=1.0):
    n = len(arr)
    freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
    fft_vals = np.fft.rfft(arr)

    mask = (freqs >= low_freq) & (freqs <= high_freq)
    fft_vals[~mask] = 0

    filtered = np.fft.irfft(fft_vals, n=n)
    return filtered
