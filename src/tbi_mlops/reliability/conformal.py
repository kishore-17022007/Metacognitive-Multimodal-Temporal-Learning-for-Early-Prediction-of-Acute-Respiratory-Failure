"""Split conformal classification (LAC score = 1 - p_true) on CALIBRATED probabilities.
Guarantee (marginal coverage >= 1-alpha) assumes exchangeability between calibration and deployment windows. It does NOT
hold under patient-level dependence (windows within a patient are correlated), distribution shift, or temporal drift."""
import numpy as np


class ConformalClassifier:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha

    def fit(self, p, y):
        y = np.asarray(y).astype(int); s = 1 - np.where(y == 1, p, 1 - p); n = len(s)
        q = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n); self.qhat = float(np.quantile(s, q, method="higher")); self.n_cal = n
        return self

    def predict_sets(self, p):
        return [[c for c, pc in ((0, 1 - pi), (1, pi)) if 1 - pc <= self.qhat] for pi in np.asarray(p, float)]

    def coverage(self, p, y):
        sets = self.predict_sets(p); return float(np.mean([int(yy) in s for s, yy in zip(sets, y)])), float(np.mean([len(s) for s in sets]))

    def to_dict(self): return {"alpha": self.alpha, "qhat": self.qhat, "n_cal": self.n_cal}

    @classmethod
    def from_dict(cls, d):
        c = cls(d["alpha"]); c.qhat, c.n_cal = d["qhat"], d["n_cal"]; return c
