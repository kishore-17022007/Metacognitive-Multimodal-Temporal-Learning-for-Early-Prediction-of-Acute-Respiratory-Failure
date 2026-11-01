"""OOD detection: Isolation Forest on window summary features (fit on train), threshold = validation-score quantile
(never test). Flags mean "unfamiliar vs training distribution", not necessarily a clinically novel state."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from ..preprocessing.features import tabular_feature_names


class OODDetector:
    def __init__(self, flag_rate: float = 0.05, seed: int = 42):
        self.flag_rate, self.seed = flag_rate, seed

    def fit(self, X_train, X_val):
        self.names, self.groups = tabular_feature_names()
        self.mu, self.sd = X_train.mean(0), X_train.std(0) + 1e-6
        self.if_ = IsolationForest(n_estimators=150, random_state=self.seed, n_jobs=1).fit(X_train)
        self.threshold = float(np.quantile(self.score(X_val), 1 - self.flag_rate))    # higher score = more anomalous
        return self

    def score(self, X):
        return -self.if_.score_samples(X)

    def flag(self, X):
        s = self.score(X); return s, s > self.threshold

    def explain(self, x, top: int = 3):
        """Which feature groups deviate most (mean |z| vs training) for one row."""
        z = np.abs((x - self.mu) / self.sd); g = {}
        for name, grp, zi in zip(self.names, self.groups, z): g.setdefault(grp, []).append(zi)
        ranked = sorted(((k, float(np.mean(v))) for k, v in g.items()), key=lambda t: -t[1])
        worst = [self.names[i] for i in np.argsort(-z)[:top]]
        return {"group_mean_abs_z": ranked, "top_features": worst}
