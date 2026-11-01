"""Loads a registered model version and serves reliability-aware predictions. Same cleaning + preprocessing as training."""
from __future__ import annotations
import json, pathlib, time
import numpy as np, pandas as pd, joblib
from ..data.schema import *
from ..preprocessing.clean import clean_observations
from ..preprocessing.pipeline import Preprocessor
from ..preprocessing.features import tabular_features
from ..reliability.conformal import ConformalClassifier
from ..reliability.assess import decide
from ..tracking import registry
from ..utils.config import ROOT, load_config

DISCLAIMER = "Research prototype on synthetic data. Risk estimates are NOT diagnoses and this is not a validated clinical service."


class ModelNotAvailable(RuntimeError):
    pass


class InputError(ValueError):
    pass


def validate_records(records: list[dict]) -> None:
    """Raises InputError for structural problems. Value-level problems (units, ranges) are handled by cleaning + flags."""
    if not records: raise InputError("no observations supplied")
    allowed = set(OBS_COLUMNS)
    for i, r in enumerate(records):
        if ID_COL not in r or TIME_COL not in r: raise InputError(f"record {i}: patient_id and timestamp are required")
        extra = set(r) - allowed
        if extra: raise InputError(f"record {i}: unknown fields {sorted(extra)}")
        if pd.isna(pd.to_datetime(r[TIME_COL], errors="coerce")): raise InputError(f"record {i}: unparseable timestamp")
    if len({r[ID_COL] for r in records}) != 1: raise InputError("all records must belong to one patient_id")


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    t = pd.to_datetime(df[TIME_COL], errors="coerce"); df[TIME_COL] = t
    if HOURS_COL not in df or df[HOURS_COL].isna().all():
        df[HOURS_COL] = (t - t.min()).dt.total_seconds() / 3600
    for c in FEATURE_NAMES:
        if c not in df: df[c] = np.nan
    df["data_source"] = df.get("data_source", "api"); df["is_synthetic"] = df.get("is_synthetic", False)
    df["data_source"] = df["data_source"].fillna("api"); df["is_synthetic"] = df["is_synthetic"].fillna(False)
    return df[OBS_COLUMNS]


class Predictor:
    def __init__(self, version: str | None = None, cfg: dict | None = None, models_dir=None):
        self.cfg = cfg or load_config(); version = version or (None if models_dir else registry.latest())
        if version is None: raise ModelNotAvailable("no trained model found - run: python scripts/train_advanced.py")
        d = (pathlib.Path(models_dir) if models_dir else ROOT / "artifacts" / "models") / version
        if not d.exists(): raise ModelNotAvailable(f"model artifacts for {version} missing")
        if not models_dir and not registry.verify(version): raise ModelNotAvailable(f"artifact checksum mismatch for {version}")
        self.version, self.dir = version, d
        self.pre = Preprocessor.load(d / "preprocessor.json"); self.ens = joblib.load(d / "ensemble.joblib"); self.cal = joblib.load(d / "calibrator.joblib")
        self.ood = joblib.load(d / "ood.joblib"); self.surv = joblib.load(d / "survival.joblib")
        self.conf = ConformalClassifier.from_dict(json.load(open(d / "conformal.json"))); self.thr = json.load(open(d / "thresholds.json"))
        self.meta = json.load(open(d / "metadata.json")); self.rcfg = self.cfg["training"]["reliability"]; self.min_obs = self.cfg["data"]["outcome"]["min_observations"]

    # model function used by the explainers: tabular features -> calibrated probability
    def model_fn(self, Xtab):
        return self.cal.predict(self.ens.member_probs(Xtab).mean(1))

    def predict_frame(self, df: pd.DataFrame) -> dict:
        t0 = time.perf_counter()
        clean, crep = clean_observations(df)
        flags = []
        if len(clean) < self.min_obs: flags.append("insufficient_history")
        if crep["steps"]["duplicates_removed"]: flags.append("duplicates_removed")
        if crep["steps"]["rows_removed_bad_timestamp"]: flags.append("timestamp_anomalies")
        if sum(crep["steps"]["invalid_set_to_missing"].values()): flags.append("invalid_values_masked")
        if sum(crep["steps"]["unit_corrections"].values()): flags.append("unit_corrections_applied")
        if clean.empty: raise InputError("no usable observations after cleaning")
        Z = self.pre.patient_steps(clean); X, valid = self.pre.window_at(Z, len(Z) - 1)
        if X[valid][:, N_F:2 * N_F].mean() < 0.45: flags.append("high_missingness")
        Xt = tabular_features(X[None].astype(np.float32), valid[None])
        A = self.ens.aggregate(Xt, self.thr["raw_threshold"]); raw = float(A["mean"][0]); p = float(self.cal.predict(A["mean"])[0])
        s, f = self.ood.flag(Xt); cset = self.conf.predict_sets([p])[0]
        r = decide(p, raw, float(A["std"][0]), float(A["agreement"][0]), self.cal.status, float(s[0]), bool(f[0]), cset, flags, self.rcfg, self.version)
        S = self.surv.survival(Xt)[0]
        ood_expl = self.ood.explain(Xt[0]) if f[0] else None
        lat = (time.perf_counter() - t0) * 1000
        return {"prediction_status": r.status, "predicted_probability": p, "risk_above_threshold": bool(p >= self.thr["calibrated_threshold"]),
                "prediction_horizon_hours": self.thr["horizon_hours"],
                "uncertainty_summary": {"ensemble_std": r.ensemble_std, "member_agreement": r.ensemble_agreement, "raw_probability": raw, "note": "model disagreement, not a calibrated interval"},
                "ood_status": {"flag": r.ood_flag, "score": r.ood_score, "threshold": self.ood.threshold, "triggers": ood_expl},
                "conformal_set": r.conformal_set, "data_quality_flags": flags, "reliability_reasons": r.reasons, "survival_curve": [float(x) for x in S],
                "model_version": self.version, "prediction_time": r.prediction_time, "inference_latency_ms": lat, "n_observations_used": int(len(clean)), "disclaimer": DISCLAIMER,
                "_window": (X, valid)}

    def predict_records(self, records: list[dict]) -> dict:
        validate_records(records); out = self.predict_frame(records_to_frame(records)); out.pop("_window"); return out
