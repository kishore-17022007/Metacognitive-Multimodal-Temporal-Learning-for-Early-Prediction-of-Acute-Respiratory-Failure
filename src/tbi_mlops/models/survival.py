"""Discrete-time survival via person-period expansion. Hazard h_k = P(event in interval k | survived to k).
S(k) = prod_{j<=k} (1 - h_j). Censored windows contribute intervals only up to their observed follow-up (event indicator 0)."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def expand(Xtab: np.ndarray, surv_time: np.ndarray, surv_event: np.ndarray, K: int):
    rows, ys = [], []
    for x, t, e in zip(Xtab, surv_time, surv_event):
        for k in range(1, int(t) + 1):
            rows.append(np.append(x, k)); ys.append(int(e == 1 and k == t))
    return np.array(rows, np.float32), np.array(ys)


class DiscreteTimeSurvival:
    def __init__(self, K: int, seed: int = 0, max_iter: int = 80):
        self.K = K
        self.clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08, max_depth=4, random_state=seed)

    def fit(self, Xtab, surv_time, surv_event):
        Xp, yp = expand(Xtab, surv_time, surv_event, self.K); self.clf.fit(Xp, yp); self.n_person_periods = len(yp); return self

    def hazards(self, Xtab):
        n = len(Xtab); out = np.zeros((n, self.K))
        for k in range(1, self.K + 1):
            out[:, k - 1] = self.clf.predict_proba(np.hstack([Xtab, np.full((n, 1), k, np.float32)]))[:, 1]
        return out

    def survival(self, Xtab):
        return np.cumprod(1.0 - self.hazards(Xtab), axis=1)   # monotone non-increasing by construction

    def risk(self, Xtab):
        return 1.0 - self.survival(Xtab)[:, -1]
