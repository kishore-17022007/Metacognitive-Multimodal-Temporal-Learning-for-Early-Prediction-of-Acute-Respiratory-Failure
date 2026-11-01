"""Uncertainty summary. Ensemble std is *model (epistemic-like) disagreement*; it is NOT separated from data noise here,
and no variance decomposition validity is claimed."""
import numpy as np


def summarize(mean, std, agreement, max_std: float):
    return [{"mean_risk": float(m), "ensemble_std": float(s), "agreement": float(a), "high_disagreement": bool(s > max_std),
             "note": "disagreement across bootstrap members; not a calibrated interval"} for m, s, a in zip(mean, std, agreement)]
