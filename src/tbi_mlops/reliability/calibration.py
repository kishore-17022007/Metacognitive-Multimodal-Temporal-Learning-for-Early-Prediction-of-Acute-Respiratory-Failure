"""Probability calibration (fit on the calibration partition only, never on test)."""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str = "sigmoid"):
        self.method = method

    def fit(self, p, y):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2 or y.sum() < 5:
            self.method, self.m = "identity", None; self.status = "identity (too few positives to calibrate)"; return self
        if self.method == "isotonic" and len(y) >= 1000 and y.sum() >= 30:
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y); self.status = "isotonic"
        else:
            self.method = "sigmoid"; self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(np.asarray(p)).reshape(-1, 1), y); self.status = "sigmoid (Platt)"
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "identity": return p
        if self.method == "isotonic": return self.m.predict(p)
        return self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
