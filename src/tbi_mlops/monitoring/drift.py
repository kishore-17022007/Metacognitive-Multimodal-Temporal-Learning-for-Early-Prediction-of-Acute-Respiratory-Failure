"""Reference-vs-current drift: PSI (decile bins from reference), KS test, missingness-rate shift. Alerts are advisory;
nothing is retrained or promoted automatically."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from ..data.schema import FEATURES


def psi(ref, cur, bins=10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float); ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 20 or len(cur) < 20: return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref); c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, dcfg: dict) -> dict:
    feats = []
    for f in FEATURES:
        p = psi(ref[f.name], cur[f.name]); a, b = ref[f.name].dropna(), cur[f.name].dropna()
        ks = ks_2samp(a, b) if len(a) > 20 and len(b) > 20 else None
        md = float(cur[f.name].isna().mean() - ref[f.name].isna().mean())
        level = "ok"
        if (p == p and p >= dcfg["psi_alert"]) or abs(md) >= dcfg["missing_rate_delta_alert"]: level = "alert"
        elif (p == p and p >= dcfg["psi_warn"]) or (ks is not None and ks.pvalue < dcfg["ks_alpha"] and p == p and p >= 0.05): level = "warn"
        feats.append({"feature": f.name, "psi": p, "ks_stat": float(ks.statistic) if ks else None, "ks_p": float(ks.pvalue) if ks else None,
                      "ref_mean": float(a.mean()) if len(a) else None, "cur_mean": float(b.mean()) if len(b) else None, "missing_rate_delta": md, "level": level})
    n_alert = sum(f["level"] == "alert" for f in feats)
    return {"features": feats, "n_alert": n_alert, "n_warn": sum(f["level"] == "warn" for f in feats), "overall": "alert" if n_alert else ("warn" if any(f["level"] == "warn" for f in feats) else "ok")}
