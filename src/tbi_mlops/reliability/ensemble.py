"""Patient-bootstrap ensemble of independently seeded gradient-boosted members (CPU deep-ensemble analogue; the PyTorch
deep ensemble in train_advanced.py follows the same aggregation interface)."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class Ensemble:
    def __init__(self, n_members=5, max_iter=80, learning_rate=0.08, max_depth=4, seed=42):
        self.n, self.kw, self.seed = n_members, dict(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth), seed
        self.members = []

    def fit(self, X, y, patient_ids):
        rng = np.random.default_rng(self.seed); pids = np.unique(patient_ids); self.members = []
        for m in range(self.n):
            bs = rng.choice(pids, len(pids), replace=True)                  # patient-level bootstrap (keeps windows grouped)
            counts = {p: c for p, c in zip(*np.unique(bs, return_counts=True))}
            w = np.array([counts.get(p, 0) for p in patient_ids], float)
            clf = HistGradientBoostingClassifier(random_state=self.seed + 101 * m, **self.kw)
            sel = w > 0
            clf.fit(X[sel], y[sel], sample_weight=w[sel]); self.members.append(clf)
        return self

    def member_probs(self, X) -> np.ndarray:
        return np.stack([m.predict_proba(X)[:, 1] for m in self.members], 1)    # (N, M)

    def aggregate(self, X, threshold: float = 0.5) -> dict:
        P = self.member_probs(X); mean = P.mean(1); std = P.std(1)
        side = (P >= threshold); agree = np.maximum(side.mean(1), 1 - side.mean(1))
        return {"mean": mean, "std": std, "agreement": agree, "members": P}
