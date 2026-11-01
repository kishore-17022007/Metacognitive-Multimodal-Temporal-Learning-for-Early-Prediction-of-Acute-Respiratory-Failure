"""Delay-aware performance monitoring: a prediction at time t only gets a label once t + horizon has passed AND follow-up
covers it. Unmatured predictions are never scored."""
from __future__ import annotations
import numpy as np, pandas as pd
from ..evaluation.classification import classification_metrics


def matured_mask(t_index_h: np.ndarray, horizon_h: float, now_h: float) -> np.ndarray:
    return (np.asarray(t_index_h) + horizon_h) <= now_h


def delayed_performance(meta: pd.DataFrame, probs: np.ndarray, threshold: float, horizon_h: float, now_h: float, mcfg: dict):
    m = matured_mask(meta.t_index_h.values, horizon_h, now_h) & meta.label_known.values
    n = int(m.sum())
    if n < mcfg["performance"]["min_matured_labels"] or len(np.unique(meta.y.values[m])) < 2:
        return {"status": "insufficient_matured_labels", "n_matured": n, "n_pending": int(len(meta) - n)}
    r = classification_metrics(meta.y.values[m], probs[m], threshold); r.update(status="ok", n_matured=n, n_pending=int(len(meta) - n)); return r


def warning_lead_time(meta: pd.DataFrame, probs, threshold, events: pd.DataFrame):
    """Hours between first threshold crossing and event, for event patients (requires event timestamps)."""
    ev = events.set_index("patient_id").event_time_h; out = []
    for pid, g in meta.assign(p=probs).groupby("patient_id"):
        if pid in ev.index and not np.isnan(ev[pid]):
            hit = g[(g.p >= threshold) & (g.t_index_h < ev[pid])]
            if len(hit): out.append(float(ev[pid] - hit.t_index_h.min()))
    return {"n_event_patients_warned": len(out), "median_lead_h": float(np.median(out)) if out else None}
