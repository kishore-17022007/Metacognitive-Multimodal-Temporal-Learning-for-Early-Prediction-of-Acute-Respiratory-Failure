"""Tabular summaries of a window for classical models, plus feature/group bookkeeping."""
import numpy as np
from ..data.schema import *


def tabular_feature_names():
    names, groups = [], []
    for s in ["last", "last_observed", "last_age", "mean", "min", "max", "slope", "n_observed"]:
        for f in FEATURES:
            names.append(f"{f.name}__{s}"); groups.append(f.group)
    for n in ["n_valid_steps", "time_since_admission", "gap_since_prev_obs"]:
        names.append(n); groups.append("temporal_context")
    return names, groups


def tabular_features(X: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """X (N,L,C), valid (N,L). Window is left-padded so the last step is the prediction index."""
    F = N_F; val, msk, age = X[:, :, :F], X[:, :, F:2 * F], X[:, :, 2 * F:3 * F]
    v3 = valid[:, :, None]; nv = np.maximum(valid.sum(1), 1)[:, None]
    mean = (val * v3).sum(1) / nv
    mn = np.where(v3, val, np.inf).min(1); mx = np.where(v3, val, -np.inf).max(1)
    first_idx = np.argmax(valid, axis=1); first = val[np.arange(len(X)), first_idx]
    slope = (val[:, -1] - first) / nv
    nobs = (msk * v3).sum(1)
    ctx = np.stack([valid.sum(1) / X.shape[1], X[:, -1, 3 * F], X[:, -1, 3 * F + 1]], 1)
    return np.concatenate([val[:, -1], msk[:, -1], age[:, -1], mean, np.where(np.isinf(mn), 0, mn), np.where(np.isinf(mx), 0, mx), slope, nobs / X.shape[1], ctx], 1).astype(np.float32)
