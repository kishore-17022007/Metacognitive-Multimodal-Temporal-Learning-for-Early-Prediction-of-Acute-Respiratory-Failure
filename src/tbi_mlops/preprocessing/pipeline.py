"""Fitted preprocessor + chronological window construction. Fit on TRAINING patients only; saved as JSON (no pickle)."""
from __future__ import annotations
import json, math
import numpy as np, pandas as pd
from ..data.schema import *
from .impute import causal_ffill

NEVER_AGE = 48.0


def n_channels() -> int:
    return N_F * 3 + 2


def channel_names() -> list[str]:
    return [f"{n}__value" for n in FEATURE_NAMES] + [f"{n}__observed" for n in FEATURE_NAMES] + [f"{n}__age" for n in FEATURE_NAMES] + ["time_since_admission", "gap_since_prev_obs"]


class Preprocessor:
    def __init__(self, window_length: int = 12):
        self.L = window_length; self.fitted = False

    def fit(self, clean_train: pd.DataFrame):
        v = clean_train[FEATURE_NAMES]
        self.median = v.median().fillna(0.0).values
        self.mean = v.mean().fillna(0.0).values
        self.std = v.std().replace(0, 1).fillna(1.0).values
        self.fitted = True
        return self

    def to_dict(self):
        return {"window_length": self.L, "features": FEATURE_NAMES, "median": self.median.tolist(), "mean": self.mean.tolist(), "std": self.std.tolist(), "never_age": NEVER_AGE}

    @classmethod
    def from_dict(cls, d):
        p = cls(d["window_length"]); p.median, p.mean, p.std = map(np.array, (d["median"], d["mean"], d["std"])); p.fitted = True
        assert d["features"] == FEATURE_NAMES, "feature schema mismatch"
        return p

    def save(self, path):
        json.dump(self.to_dict(), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.load(open(path)))

    def patient_steps(self, g: pd.DataFrame) -> np.ndarray:
        """Per-observation channel matrix (n, C). Row i depends only on rows <= i."""
        g = g.sort_values(TIME_COL, kind="mergesort")
        t = g[HOURS_COL].values.astype(float); V = g[FEATURE_NAMES].values.astype(float)
        filled, mask, age = causal_ffill(V, t, self.median, NEVER_AGE)
        gap = np.diff(t, prepend=t[0])
        return np.concatenate([(filled - self.mean) / self.std, mask, np.log1p(age) / math.log1p(NEVER_AGE),
                               (t / 48.0)[:, None], (np.log1p(gap) / 2.0)[:, None]], axis=1)

    def window_at(self, Z: np.ndarray, i: int):
        lo = max(0, i - self.L + 1); w = Z[lo:i + 1]
        X = np.zeros((self.L, Z.shape[1])); valid = np.zeros(self.L, bool)
        X[self.L - len(w):] = w; valid[self.L - len(w):] = True      # left-padded
        return X, valid


def build_dataset(clean: pd.DataFrame, patients: pd.DataFrame, pre: Preprocessor, ocfg: dict) -> dict:
    """Prediction index = each `stride`-th observation. See OUTCOME_DEFINITION. Only past observations enter X."""
    H, K = ocfg["horizon_hours"], ocfg["n_intervals"]; w = H / K
    pat = patients.set_index("patient_id")
    Xs, Vs, rows = [], [], []
    for pid, g in clean.groupby(ID_COL, sort=False):
        if pid not in pat.index: continue
        p = pat.loc[pid]
        if not bool(p.outcome_known): continue                     # unknown outcome -> excluded (documented)
        g = g.sort_values(TIME_COL, kind="mergesort"); Z = pre.patient_steps(g); t = g[HOURS_COL].values
        ev, T, fe = bool(p.event_observed), float(p.event_time_h), float(p.followup_end_h)
        for i in range(ocfg["min_observations"] - 1, len(g), ocfg["stride"]):
            ti = float(t[i])
            if ev and T <= ti: continue                              # event already happened -> not a prediction time
            rem = (T - ti) if ev else math.inf
            horizon_covered = ev or (ti + H <= fe)
            label_known = horizon_covered
            y = int(ev and rem <= H)
            if ev and rem <= H:
                st, se = int(math.ceil(rem / w)), 1
            else:
                fu = H if (ev or ti + H <= fe) else (fe - ti)
                st, se = int(math.floor(fu / w + 1e-9)), 0
                if st < 1: continue
            kk = np.arange(1, K + 1); cum = ((rem <= kk * w) if ev else np.zeros(K, bool)).astype(float)
            tv = (((kk * w) <= (fe - ti)) | cum.astype(bool)).astype(float) if not ev else np.ones(K)
            X, valid = pre.window_at(Z, i)
            Xs.append(X); Vs.append(valid)
            rows.append(dict(patient_id=pid, t_index_h=ti, obs_index=i, y=y, label_known=label_known, surv_time=st, surv_event=se,
                             **{f"traj_{k}": cum[k] for k in range(K)}, **{f"trajvalid_{k}": tv[k] for k in range(K)}))
    meta = pd.DataFrame(rows)
    return {"X": np.stack(Xs).astype(np.float32), "valid": np.stack(Vs), "meta": meta, "K": K, "H": H}


def subset(ds: dict, mask) -> dict:
    mask = np.asarray(mask)
    return {"X": ds["X"][mask], "valid": ds["valid"][mask], "meta": ds["meta"][mask].reset_index(drop=True), "K": ds["K"], "H": ds["H"]}


def split_patients(patients: pd.DataFrame, fractions: dict, seed: int) -> dict:
    """Patient-level, event-stratified split. Returns {partition: set(patient_id)}."""
    rng = np.random.default_rng(seed); parts = {k: [] for k in fractions}; names = list(fractions)
    for ev, g in patients.groupby("event_observed"):
        ids = np.array(g.patient_id.tolist()); rng.shuffle(ids)
        cuts = np.round(np.cumsum([fractions[k] for k in names]) * len(ids)).astype(int); cuts[-1] = len(ids)
        start = 0
        for k, c in zip(names, cuts):
            parts[k] += list(ids[start:c]); start = c
    return {k: set(v) for k, v in parts.items()}
