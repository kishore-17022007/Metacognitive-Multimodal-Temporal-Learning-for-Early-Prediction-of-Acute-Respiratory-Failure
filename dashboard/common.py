import json, pathlib
import pandas as pd, streamlit as st
from tbi_mlops.utils.config import ROOT, load_config

REPORTS = ROOT / "artifacts" / "reports"
GRAPHS = ROOT / "artifacts" / "graphs"


def report(name):
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def need(obj, hint="python scripts/train_baseline.py && python scripts/train_advanced.py"):
    if obj is None:
        st.info(f"No results yet. Run `{hint}` first."); st.stop()
    return obj


@st.cache_resource
def get_predictor():
    from tbi_mlops.serving.predictor import Predictor, ModelNotAvailable
    try: return Predictor(cfg=load_config())
    except ModelNotAvailable as e: return str(e)


@st.cache_data
def clean_obs():
    for n, rd in (("clean_observations.csv", pd.read_csv), ("clean_observations.parquet", pd.read_parquet)):
        p = ROOT / "data" / "processed" / n
        if p.exists():
            d = rd(p); d["timestamp"] = pd.to_datetime(d["timestamp"]); return d
    return None


@st.cache_data
def patients():
    p = ROOT / "data" / "synthetic" / "patients.csv"; return pd.read_csv(p) if p.exists() else None
