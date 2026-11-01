"""Cleaning: correctable errors are corrected (units), uncorrectable ones become NaN (never invented). Input is never mutated."""
from __future__ import annotations
import numpy as np, pandas as pd
from ..data.schema import *


def quality_snapshot(df: pd.DataFrame) -> dict:
    """Counts on a frame whose feature columns may still be object dtype. Missing columns are treated as all-missing."""
    df = df.copy()
    for f in FEATURES:
        if f.name not in df: df[f.name] = np.nan
    for c in (ID_COL, TIME_COL):
        if c not in df: df[c] = np.nan
    num = {f.name: pd.to_numeric(df[f.name], errors="coerce") for f in FEATURES}
    snap = {"rows": int(len(df)), "columns": int(df.shape[1]),
            "missing_per_variable": {k: int(v.isna().sum()) for k, v in num.items()},
            "non_numeric_tokens": {f.name: int((pd.to_numeric(df[f.name], errors="coerce").isna() & df[f.name].notna() & (df[f.name].astype(str).str.lower() != "nan")).sum()) for f in FEATURES},
            "invalid_per_variable": {f.name: int(((v < f.lo) | (v > f.hi)).sum()) for f, v in zip(FEATURES, num.values())},
            "duplicate_patient_time": int(df.duplicated([ID_COL, TIME_COL]).sum())}
    ts = pd.to_datetime(df[TIME_COL], errors="coerce")
    snap["unparseable_timestamps"] = int(ts.isna().sum())
    return snap


def _outlier_counts(df):
    out = {}
    for f in FEATURES:
        if not f.continuous: continue
        v = df[f.name].dropna()
        if len(v) < 20: out[f.name] = 0; continue
        med = v.median(); mad = (v - med).abs().median() * 1.4826 or 1e-9
        out[f.name] = int((((df[f.name] - med).abs() / mad) > 6).sum())
    return out


def clean_observations(raw: pd.DataFrame, flag_outliers_only: bool = True):
    """Returns (clean_df, report). Outliers are *flagged in the report*, not removed (clinically possible values)."""
    before = quality_snapshot(raw)
    df = raw.copy()
    rep = {"before": before, "steps": {}}
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    n0 = len(df); df = df.dropna(subset=[TIME_COL, ID_COL]); rep["steps"]["rows_removed_bad_timestamp"] = n0 - len(df)
    for f in FEATURES:
        df[f.name] = pd.to_numeric(df[f.name], errors="coerce")
    df[HOURS_COL] = pd.to_numeric(df[HOURS_COL], errors="coerce")
    df = df.sort_values([ID_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    d = df.duplicated([ID_COL, TIME_COL], keep="first")
    conflicting = 0
    if d.any():  # a duplicate is "conflicting" if any feature differs from the kept row
        kept = df[~d].set_index([ID_COL, TIME_COL])[FEATURE_NAMES]
        dd = df[d].set_index([ID_COL, TIME_COL])[FEATURE_NAMES]
        k = kept.loc[dd.index.unique()] if len(dd) else kept.iloc[:0]
        for (idx, row) in dd.iterrows():
            ref = kept.loc[idx]
            diff = ~((ref == row) | (ref.isna() & row.isna()))
            conflicting += int(diff.any())
    rep["steps"]["duplicates_removed"] = int(d.sum()); rep["steps"]["conflicting_duplicates_kept_first"] = conflicting
    df = df[~d].reset_index(drop=True)
    units = {}
    for col, (cond, fn) in {"spo2": (lambda s: (s > 0) & (s <= 1.0), lambda s: s * 100),
                            "fio2": (lambda s: (s > 1.0) & (s <= 100), lambda s: s / 100),
                            "paco2": (lambda s: (s > 0) & (s < 15), lambda s: s * 7.5),
                            "hemoglobin": (lambda s: s > 30, lambda s: s / 10)}.items():
        m = cond(df[col]); units[col] = int(m.sum()); df.loc[m, col] = fn(df.loc[m, col])
    rep["steps"]["unit_corrections"] = units
    inval = {}
    for f in FEATURES:
        m = (df[f.name] < f.lo) | (df[f.name] > f.hi); inval[f.name] = int(m.sum()); df.loc[m, f.name] = np.nan
    rep["steps"]["invalid_set_to_missing"] = inval
    rep["steps"]["outlier_flags_not_removed"] = _outlier_counts(df)
    df["data_source"] = df["data_source"].astype(str)
    df["is_synthetic"] = df["is_synthetic"].astype(str).str.lower().isin(["true", "1"])
    after = {"rows": int(len(df)), "columns": int(df.shape[1]),
             "missing_per_variable": {f.name: int(df[f.name].isna().sum()) for f in FEATURES},
             "duplicate_patient_time": int(df.duplicated([ID_COL, TIME_COL]).sum()),
             "invalid_per_variable": {f.name: int(((df[f.name] < f.lo) | (df[f.name] > f.hi)).sum()) for f in FEATURES}}
    rep["after"] = after
    rep["unresolved"] = {"note": "Remaining missing values are genuine/unrecoverable and are modelled with masks; extreme outliers retained but flagged.",
                         "total_missing_cells_after": int(sum(after["missing_per_variable"].values()))}
    return df, rep
