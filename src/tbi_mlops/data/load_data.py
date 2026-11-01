"""Loaders. Synthetic loader + an adapter for APPROVED, de-identified research data (e.g. MIMIC-IV).

MIMIC-IV requires PhysioNet credentialing and a signed data-use agreement. This code never downloads data and
never fabricates observations: unmapped internal features are returned as NaN and listed as unavailable.
"""
from __future__ import annotations
import pandas as pd, yaml
from .schema import *
from ..utils.config import path


def load_synthetic():
    obs = pd.read_csv(path("data", "synthetic", "raw_observations.csv"), dtype=object)  # keep raw tokens
    pat = pd.read_csv(path("data", "synthetic", "patients.csv"))
    return obs, pat


class RealDataAdapter:
    """mapping YAML example:
    observations_csv: /secure/path/chartevents_long.csv   # columns: subject_id, charttime, itemid, valuenum
    patient_id: subject_id
    time: charttime
    items: {heart_rate: [220045], spo2: [220277], ...}      # internal feature -> source item ids
    outcomes_csv: /secure/path/outcomes.csv                  # clinician-adjudicated ARF labels REQUIRED
    """
    def __init__(self, mapping_yaml: str):
        self.m = yaml.safe_load(open(mapping_yaml))

    def unavailable_features(self):
        return [f for f in FEATURE_NAMES if f not in self.m.get("items", {})]

    def load(self):
        raw = pd.read_csv(self.m["observations_csv"])
        raw[TIME_COL] = pd.to_datetime(raw[self.m["time"]], errors="coerce")
        inv = {i: f for f, ids in self.m["items"].items() for i in ids}
        raw["feature"] = raw[self.m.get("item_col", "itemid")].map(inv)
        raw = raw.dropna(subset=["feature", TIME_COL])
        wide = raw.pivot_table(index=[self.m["patient_id"], TIME_COL], columns="feature", values=self.m.get("value_col", "valuenum"), aggfunc="first").reset_index()
        wide = wide.rename(columns={self.m["patient_id"]: ID_COL})
        for f in self.unavailable_features():
            wide[f] = float("nan")                                   # unavailable, NOT fabricated
        wide[HOURS_COL] = (wide[TIME_COL] - wide.groupby(ID_COL)[TIME_COL].transform("min")).dt.total_seconds() / 3600
        wide["data_source"], wide["is_synthetic"] = self.m.get("name", "real"), False
        wide[TIME_COL] = wide[TIME_COL].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return wide[OBS_COLUMNS]
