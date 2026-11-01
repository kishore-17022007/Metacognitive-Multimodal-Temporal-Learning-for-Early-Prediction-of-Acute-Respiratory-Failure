"""Baseline attribution. SHAP is not installed in the build sandbox -> permutation importance (global) and a
reference-substitution local attribution are used. If `shap` is available, shap_values() is used for tree models."""
from __future__ import annotations
import numpy as np
from sklearn.inspection import permutation_importance


def global_importance(model, X, y, names, groups, seed=0, n_repeats=5):
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=seed, n_jobs=1)
    feats = sorted(zip(names, r.importances_mean, r.importances_std), key=lambda t: -t[1])
    g = {}
    for n, grp, m in zip(names, groups, r.importances_mean): g[grp] = g.get(grp, 0) + max(m, 0)
    return {"method": "permutation_importance(avg_precision, validation)", "features": [{"name": n, "importance": float(m), "std": float(s)} for n, m, s in feats],
            "group_importance": {k: float(v) for k, v in sorted(g.items(), key=lambda t: -t[1])}}


def shap_values_if_available(model, X):
    try:
        import shap
        return shap.TreeExplainer(model).shap_values(X)
    except Exception:
        return None


def local_attribution(predict_fn, x, reference, names, top=10):
    """Delta in risk when each feature is replaced by its reference (training-median) value. Not causal."""
    base = float(predict_fn(x[None])[0]); out = []
    for j, n in enumerate(names):
        z = x.copy(); z[j] = reference[j]; out.append((n, base - float(predict_fn(z[None])[0])))
    return base, sorted(out, key=lambda t: -abs(t[1]))[:top]
