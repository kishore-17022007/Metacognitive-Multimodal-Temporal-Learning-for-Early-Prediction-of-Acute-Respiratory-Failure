"""Chronological replay source + synthetic scenario perturbations (applied causally to each incoming observation)."""
from __future__ import annotations
import numpy as np, pandas as pd
from ..data.schema import *

SCENARIOS = ["normal", "increasing_instability", "missing_data", "abnormal_inputs", "gradual_drift", "abrupt_drift", "high_latency"]
ONSET_OBS = 8   # scenario perturbation starts after this many observations


def perturb(row: dict, scenario: str, k: int, rng) -> tuple[dict, float]:
    """k = observations since onset (>=0). Returns (row, extra_simulated_latency_ms). Never reads future data."""
    r = dict(row); extra = 0.0
    if scenario == "normal" or k < 0: return r, extra
    c = lambda v, lo, hi: float(np.clip(v, lo, hi))
    if scenario == "increasing_instability":
        r["resp_rate"] = c(r["resp_rate"] + 0.9 * k, 4, 60); r["spo2"] = c(r["spo2"] - 0.6 * k, 50, 100); r["heart_rate"] = c(r["heart_rate"] + 1.5 * k, 20, 250)
        r["o2_support"] = c(r["o2_support"] + k // 4, 0, 3); r["fio2"] = c(r["fio2"] + 0.04 * k, 0.21, 1.0)
    elif scenario == "missing_data":
        for f in FEATURE_NAMES:
            if GROUP_OF[f] in ("respiratory", "cardiovascular") and rng.random() < 0.8: r[f] = np.nan
    elif scenario == "abnormal_inputs":
        r.update(resp_rate=52.0, heart_rate=205.0, sbp=215.0, lactate=22.0, spo2=72.0, hemoglobin=4.5)
    elif scenario == "gradual_drift":
        r["resp_rate"] = c(r["resp_rate"] + 0.5 * k, 4, 60); r["heart_rate"] = c(r["heart_rate"] + 0.8 * k, 20, 250); r["hemoglobin"] = c(r["hemoglobin"] - 0.2 * k, 3, 22) if r["hemoglobin"] == r["hemoglobin"] else r["hemoglobin"]
    elif scenario == "abrupt_drift":
        r["spo2"] = c(r["spo2"] - 12, 50, 100); r["heart_rate"] = c(r["heart_rate"] + 40, 20, 250); r["map"] = c(r["map"] - 25, 30, 180)
    elif scenario == "high_latency":
        extra = 400.0 + float(rng.uniform(0, 100))
    return r, extra


class Replay:
    """Observations of ONE patient, kept in original chronological order; never reordered."""
    def __init__(self, patient_obs: pd.DataFrame):
        self.obs = patient_obs.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)
        assert self.obs[HOURS_COL].is_monotonic_increasing
    def __len__(self): return len(self.obs)
    def row(self, i) -> dict: return self.obs.iloc[i].to_dict()
    def hour(self, i) -> float: return float(self.obs[HOURS_COL].iloc[i])
