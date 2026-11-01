"""Cohort-level synthetic drift simulation: perturb test patients' observations after onset, then measure PSI/KS drift vs the
TRAIN reference, OOD rate, mean risk, ensemble disagreement, and latency. Scenarios are SYNTHETIC."""
from __future__ import annotations
import time
import numpy as np, pandas as pd
from ..data.schema import *
from ..streaming.replay import perturb, ONSET_OBS, SCENARIOS
from ..preprocessing.features import tabular_features
from ..monitoring.drift import drift_report


def perturb_frame(clean: pd.DataFrame, scenario: str, seed: int = 0):
    rng = np.random.default_rng(seed); rows, extra = [], []
    for pid, g in clean.groupby(ID_COL, sort=False):
        g = g.sort_values(TIME_COL, kind="mergesort")
        for i, r in enumerate(g.to_dict("records")):
            r2, ex = perturb(r, scenario, i - ONSET_OBS if i >= ONSET_OBS else -1, rng); rows.append(r2); extra.append(ex)
    return pd.DataFrame(rows), float(np.mean(extra) if extra else 0)


def run_cohort(pred, clean_test: pd.DataFrame, reference: pd.DataFrame, mcfg: dict, scenarios=SCENARIOS, n_patients=60, seed=0):
    ids = clean_test[ID_COL].drop_duplicates().sample(min(n_patients, clean_test[ID_COL].nunique()), random_state=seed)
    sub = clean_test[clean_test[ID_COL].isin(ids)]; out = {}
    for sc in scenarios:
        cur, extra_ms = perturb_frame(sub, sc, seed); t0 = time.perf_counter(); Xs = []
        for pid, g in cur.groupby(ID_COL, sort=False):
            Z = pred.pre.patient_steps(g)
            for i in range(ONSET_OBS + 2, len(Z), 2):                      # windows after onset
                Xs.append(pred.pre.window_at(Z, i))
        if not Xs: continue
        X = np.stack([a for a, _ in Xs]).astype(np.float32); V = np.stack([b for _, b in Xs]); T = tabular_features(X, V)
        A = pred.ens.aggregate(T, pred.thr["raw_threshold"]); s, f = pred.ood.flag(T); lat = (time.perf_counter() - t0) / len(T) * 1000 + extra_ms
        post = cur.groupby(ID_COL, sort=False).apply(lambda g: g.sort_values(TIME_COL).iloc[ONSET_OBS:], include_groups=False)
        d = drift_report(reference, post, mcfg["drift"])
        out[sc] = {"n_windows": int(len(T)), "mean_risk": float(pred.cal.predict(A["mean"]).mean()), "mean_ensemble_std": float(A["std"].mean()), "ood_rate": float(f.mean()),
                   "drift_overall": d["overall"], "n_features_alert": d["n_alert"], "n_features_warn": d["n_warn"], "avg_latency_ms_per_window": lat,
                   "top_drifted": [x["feature"] for x in sorted(d["features"], key=lambda x: -(x["psi"] if x["psi"] == x["psi"] else 0))[:3]]}
    return out
