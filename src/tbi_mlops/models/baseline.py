from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_baselines(mcfg: dict, seed: int = 42) -> dict:
    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=mcfg["logistic"]["C"], max_iter=mcfg["logistic"]["max_iter"], class_weight="balanced")),
        "random_forest": RandomForestClassifier(**mcfg["random_forest"], class_weight="balanced_subsample", n_jobs=1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=mcfg["ensemble"]["max_iter"], learning_rate=mcfg["ensemble"]["learning_rate"], max_depth=mcfg["ensemble"]["max_depth"], random_state=seed),
    }
    try:
        import xgboost as xgb
        models["xgboost"] = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, n_jobs=1, random_state=seed, eval_metric="logloss")
    except ImportError:
        pass   # optional; documented fallback = hist_gradient_boosting
    return models
