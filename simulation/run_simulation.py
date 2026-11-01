"""Headless simulation: cohort-level drift scenarios + one streamed patient. Usage: python simulation/run_simulation.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import json
import pandas as pd
from tbi_mlops.utils.config import load_config, path
from tbi_mlops.serving.predictor import Predictor
from tbi_mlops.evaluation.simulate_cohort import run_cohort
from tbi_mlops.evaluation.reporting import write_json
from tbi_mlops.streaming.engine import StreamEngine

cfg = load_config(); P = Predictor(cfg=cfg)
clean = pd.read_csv(path("data", "processed", "clean_observations.csv")); clean["timestamp"] = pd.to_datetime(clean["timestamp"])
splits = json.load(open(path("data", "processed", "splits.json")))
ref = clean[clean.patient_id.isin(splits["train"])]; test = clean[clean.patient_id.isin(splits["test"])]
res = run_cohort(P, test, ref, cfg["monitoring"])
write_json(res, "drift_simulation.json")
print(f"{'scenario':24s} {'drift':6s} {'#alert':>6s} {'OOD%':>6s} {'risk':>6s} {'std':>6s} {'ms/win':>7s}")
for k, v in res.items():
    print(f"{k:24s} {v['drift_overall']:6s} {v['n_features_alert']:6d} {100*v['ood_rate']:6.1f} {v['mean_risk']:6.3f} {v['mean_ensemble_std']:6.3f} {v['avg_latency_ms_per_window']:7.1f}")
pat = pd.read_csv(path("data", "synthetic", "patients.csv")); cnt = clean.groupby("patient_id").size()
ev = pat[pat.event_observed & pat.outcome_known & pat.patient_id.isin(splits["test"])].assign(n=lambda d: d.patient_id.map(cnt)).sort_values("n").iloc[-1]
e = StreamEngine(P, clean[clean.patient_id == ev.patient_id], "normal", ev.event_time_h, reference=ref, monitoring_cfg=cfg["monitoring"]); e.start()
while not e.finished: e.tick(1, 3600)
print(f"[stream] patient streamed: {len(e.predictions)} predictions, {len(e.alert_history)} alerts, ops={json.dumps({k: round(v,1) if isinstance(v,float) else v for k,v in e.ops.summary().items()})}")
