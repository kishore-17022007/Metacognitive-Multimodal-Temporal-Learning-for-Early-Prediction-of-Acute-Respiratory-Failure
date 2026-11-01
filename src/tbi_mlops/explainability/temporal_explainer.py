"""Temporal occlusion attribution for any window->probability function.
Occluding step s = replace that step by 'not observed' (zeros in padded/valid=False), so features are recomputed without it.
Attribution = baseline_risk - occluded_risk. Descriptive only: not causal, and unstable across windows."""
from __future__ import annotations
import numpy as np
from ..data.schema import *
from ..preprocessing.features import tabular_features


def _predict(model_fn, X, valid):
    return model_fn(tabular_features(X, valid))


def step_attribution(model_fn, x, valid):
    """x (L,C), valid (L,). Returns array (L,) with attributions for valid steps (0 for padding)."""
    X, V = x[None].copy(), valid[None].copy(); base = float(_predict(model_fn, X, V)[0]); att = np.zeros(len(valid))
    for s in np.where(valid)[0]:
        Xo, Vo = X.copy(), V.copy(); Xo[0, s] = 0; Vo[0, s] = False
        att[s] = base - float(_predict(model_fn, Xo, Vo)[0])
    return base, att


def temporal_feature_attribution(model_fn, x, valid):
    """Return risk change when each feature at each valid step is occluded."""
    X, V = x[None].copy(), valid[None].copy()
    base = float(_predict(model_fn, X, V)[0])
    attribution = np.zeros((len(valid), N_F), dtype=float)
    for step in np.where(valid)[0]:
        for feature in range(N_F):
            occluded = X.copy()
            occluded[0, step, feature] = 0
            occluded[0, step, N_F + feature] = 0
            occluded[0, step, 2 * N_F + feature] = 1
            attribution[step, feature] = base - float(_predict(model_fn, occluded, V)[0])
    return base, attribution


def temporal_differences(x, valid):
    """Return adjacent observed value changes in preprocessed feature units."""
    changes = np.full((len(valid), N_F), np.nan, dtype=float)
    steps = np.where(valid)[0]
    for previous, current in zip(steps[:-1], steps[1:]):
        both_observed = (x[previous, N_F:2 * N_F] > 0) & (x[current, N_F:2 * N_F] > 0)
        changes[current, both_observed] = x[current, :N_F][both_observed] - x[previous, :N_F][both_observed]
    return changes


def group_attribution(model_fn, x, valid, medians_scaled=None):
    """Occlude all value/mask/age channels of a feature group in the whole window (set to 0 = scaled median)."""
    X, V = x[None].copy(), valid[None]; base = float(_predict(model_fn, X, V)[0]); out = {}
    for g in GROUPS:
        Xo = X.copy()
        for j, f in enumerate(FEATURE_NAMES):
            if GROUP_OF[f] == g:
                Xo[0, :, j] = 0; Xo[0, :, N_F + j] = 0; Xo[0, :, 2 * N_F + j] = 1.0   # value=median, unobserved, max age
        out[g] = base - float(_predict(model_fn, Xo, V)[0])
    return base, out


def data_quality_notes(x, valid):
    """Reported SEPARATELY from predictive contributions."""
    msk = x[valid][:, N_F:2 * N_F]; notes = []
    miss = 1 - msk.mean(0) if len(msk) else np.ones(N_F)
    for f, m in zip(FEATURE_NAMES, miss):
        if m > 0.8: notes.append(f"{f}: {m:.0%} of steps unobserved in window")
    return notes
