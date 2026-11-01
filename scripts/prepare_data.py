import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import argparse, json
from tbi_mlops.utils.config import load_config
from tbi_mlops.data.dataset import prepare
ap = argparse.ArgumentParser(); ap.add_argument("--profile", default=None); a = ap.parse_args()
cfg = load_config(a.profile); out = prepare(cfg)
r = json.load(open(__import__("tbi_mlops.utils.config", fromlist=["path"]).path("artifacts", "reports", "data_quality_report.json")))
print("[prepare] partitions:", json.dumps(r["partition_stats"]))
print("[prepare] steps:", json.dumps({k: v for k, v in r["clean_report"]["steps"].items() if k != "outlier_flags_not_removed"}))
