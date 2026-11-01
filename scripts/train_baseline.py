import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import argparse
from tbi_mlops.utils.config import load_config
from tbi_mlops.training import train_baselines
ap = argparse.ArgumentParser(); ap.add_argument("--profile", default=None); a = ap.parse_args()
r = train_baselines(load_config(a.profile))
for k, v in r.items(): print(f"{k:24s} test AUROC={v['test']['auroc']:.3f} AUPRC={v['test']['auprc']:.3f} F1={v['test']['f1']:.3f} Brier={v['test']['brier']:.4f} lat_p50={v['latency']['p50_ms']:.2f}ms")
