"""End-to-end data preparation: generate/load -> clean -> split by patient -> fit preprocessor on TRAIN -> windows."""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from .generate_synthetic import generate
from .load_data import load_synthetic
from .validate_data import validate_clean
from ..preprocessing.clean import clean_observations
from ..preprocessing.pipeline import Preprocessor, build_dataset, split_patients, subset
from ..utils.config import path

DATASET_VERSION_FILE = "dataset_manifest.json"


def prepare(cfg: dict, save: bool = True, raw=None, pat=None) -> dict:
    dcfg = cfg["data"]
    if raw is None or pat is None:
        try:
            raw, pat = load_synthetic()
        except FileNotFoundError:
            raw, pat, _ = generate(dcfg)
    raw_copy_hash = int(pd.util.hash_pandas_object(raw.astype(str), index=False).sum() % (1 << 31))   # original stays untouched
    clean, crep = clean_observations(raw)
    errs = validate_clean(clean)
    if errs: raise ValueError(f"cleaned data failed validation: {errs}")
    splits = split_patients(pat, dcfg["split"]["fractions"], dcfg["seed"])
    pre = Preprocessor(dcfg["outcome"]["window_length"]).fit(clean[clean.patient_id.isin(splits["train"])])   # TRAIN ONLY
    ds = build_dataset(clean, pat, pre, dcfg["outcome"])
    parts = {k: subset(ds, ds["meta"].patient_id.isin(v).values) for k, v in splits.items()}
    out = {"clean": clean, "patients": pat, "splits": splits, "pre": pre, "ds": ds, "parts": parts, "clean_report": crep, "raw_hash": raw_copy_hash}
    if save:
        pre.save(path("artifacts", "preprocessors", "preprocessor.json"))
        np.savez_compressed(path("data", "processed", "windows.npz"), X=ds["X"], valid=ds["valid"])
        ds["meta"].to_csv(path("data", "processed", "windows_meta.csv"), index=False)
        clean.to_parquet(path("data", "processed", "clean_observations.parquet")) if _has_pyarrow() else clean.to_csv(path("data", "processed", "clean_observations.csv"), index=False)
        json.dump({k: sorted(v) for k, v in splits.items()}, open(path("data", "processed", "splits.json"), "w"))
        stats = {k: {"windows": int(len(p["meta"])), "patients": int(p["meta"].patient_id.nunique()), "positive_rate": float(p["meta"].y[p["meta"].label_known].mean()) if len(p["meta"]) else None} for k, p in parts.items()}
        rep = {"clean_report": crep, "partition_stats": stats, "n_windows": int(len(ds["meta"])), "channels": int(ds["X"].shape[2]), "window_length": int(ds["X"].shape[1]), "raw_data_hash": raw_copy_hash,
               "dataset_version": f"synthetic-{raw_copy_hash}", "synthetic": True}
        json.dump(rep, open(path("artifacts", "reports", "data_quality_report.json"), "w"), indent=2, default=str)
    return out


def _has_pyarrow():
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        return False


def known(part: dict):
    """Windows with a known binary label."""
    m = part["meta"].label_known.values
    return subset(part, m)
