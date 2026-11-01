"""Explicit schema validation (Pandera-free)."""
import pandas as pd
from .schema import *


def validate_raw_schema(df: pd.DataFrame) -> list[str]:
    errs = [f"missing column: {c}" for c in OBS_COLUMNS if c not in df.columns]
    if not errs and df.empty:
        errs.append("empty dataframe")
    return errs


def validate_clean(df: pd.DataFrame) -> list[str]:
    errs = validate_raw_schema(df)
    if errs:
        return errs
    if not pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
        errs.append("timestamp not datetime")
    for f in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[f.name]):
            errs.append(f"{f.name} not numeric"); continue
        v = df[f.name].dropna()
        if ((v < f.lo) | (v > f.hi)).any():
            errs.append(f"{f.name} has out-of-range values")
    if df.duplicated([ID_COL, TIME_COL]).any():
        errs.append("duplicate patient-time rows")
    for _, g in df.groupby(ID_COL):
        if not g[TIME_COL].is_monotonic_increasing:
            errs.append("timestamps not chronological"); break
    return errs
