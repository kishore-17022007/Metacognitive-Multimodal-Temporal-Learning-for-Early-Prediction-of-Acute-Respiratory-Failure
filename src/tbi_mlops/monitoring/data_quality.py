import pandas as pd
from ..data.schema import *
from ..preprocessing.clean import quality_snapshot


def batch_quality(raw: pd.DataFrame, mcfg: dict) -> dict:
    """Quality of an incoming RAW batch (before cleaning)."""
    snap = quality_snapshot(raw); n = max(snap["rows"], 1); cells = n * N_F
    miss = sum(snap["missing_per_variable"].values()) / cells
    tokens = sum(snap["non_numeric_tokens"].values()) / cells
    inval = sum(snap["invalid_per_variable"].values()) / cells
    dup = snap["duplicate_patient_time"] / n
    schema = [c for c in OBS_COLUMNS if c not in raw.columns]
    t = pd.to_datetime(raw[TIME_COL], errors="coerce") if TIME_COL in raw else pd.Series(dtype="datetime64[ns]")
    flags = []
    th = mcfg["data_quality"]
    if schema: flags.append("schema_violation")
    if miss > th["max_missing_rate"]: flags.append("high_missingness")
    if inval + tokens > th["max_invalid_rate"]: flags.append("high_invalid_rate")
    if dup > th["max_duplicate_rate"]: flags.append("duplicates")
    if snap["unparseable_timestamps"]: flags.append("timestamp_anomalies")
    return {"rows": snap["rows"], "missing_rate": miss, "invalid_rate": inval, "non_numeric_rate": tokens, "duplicate_rate": dup,
            "unparseable_timestamps": snap["unparseable_timestamps"], "schema_missing_columns": schema, "flags": flags}
