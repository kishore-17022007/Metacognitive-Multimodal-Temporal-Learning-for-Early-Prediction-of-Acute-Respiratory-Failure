"""Deterministic small fixtures (no real artifacts touched). Built once per process."""
import sys, pathlib, tempfile, copy, functools
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from tbi_mlops.utils.config import load_config
from tbi_mlops.data.generate_synthetic import generate
from tbi_mlops.data.dataset import prepare
from tbi_mlops.training import train_reliability_stack
from tbi_mlops.serving.predictor import Predictor


def tiny_cfg():
    cfg = copy.deepcopy(load_config("dev")); cfg["data"]["n_patients"] = 250; cfg["data"]["seed"] = 7
    cfg["model"]["ensemble"].update(n_members=2, max_iter=25); return cfg


@functools.lru_cache(maxsize=1)
def tiny():
    cfg = tiny_cfg(); raw, pat, rep = generate(cfg["data"]); D = prepare(cfg, save=False, raw=raw, pat=pat)
    tmp = tempfile.mkdtemp(prefix="tbi_models_"); out = train_reliability_stack(cfg, D, version="vtest", models_root=tmp, persist=False)
    P = Predictor("vtest", cfg=cfg, models_dir=tmp)
    return dict(cfg=cfg, raw=raw, pat=pat, rep=rep, D=D, out=out, P=P, models_dir=tmp)


def long_patient(kind="nonevent"):
    t = tiny(); clean, pat = t["D"]["clean"], t["pat"]; cnt = clean.groupby("patient_id").size()
    q = pat[(pat.event_observed == (kind == "event")) & pat.outcome_known].assign(n=lambda d: d.patient_id.map(cnt)).sort_values("n").iloc[-1]
    return clean[clean.patient_id == q.patient_id], q
