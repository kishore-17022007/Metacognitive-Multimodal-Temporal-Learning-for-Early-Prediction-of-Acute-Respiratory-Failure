"""Causal imputation: forward-fill within patient (past only), then training-median. Observation-age is tracked."""
import numpy as np


def causal_ffill(V: np.ndarray, t: np.ndarray, medians: np.ndarray, never_age: float = 48.0):
    """V (n,F) with NaN. Returns (filled, mask, age_hours). Uses only past/current rows -> leakage-safe."""
    n, F = V.shape
    filled = np.empty_like(V); age = np.zeros_like(V); mask = (~np.isnan(V)).astype(float)
    last = np.full(F, np.nan); last_t = np.full(F, np.nan)
    for i in range(n):
        obs = mask[i] == 1
        last[obs] = V[i, obs]; last_t[obs] = t[i]
        filled[i] = np.where(np.isnan(last), medians, last)
        age[i] = np.where(np.isnan(last_t), never_age, t[i] - last_t)
    return filled, mask, age
