"""Train the reliability stack locally or publish an AWS candidate bundle."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import argparse, json
from tbi_mlops.utils.config import load_config
from tbi_mlops.training import train_reliability_stack
ap = argparse.ArgumentParser(); ap.add_argument("--profile", default=None); ap.add_argument("--version", default=None); ap.add_argument("--skip-torch", action="store_true"); ap.add_argument("--aws", action="store_true", help="Download S3 inputs and publish a scikit-learn candidate bundle"); a = ap.parse_args()
cfg = load_config(a.profile)
if a.aws:
    if not a.skip_torch: ap.error("--aws requires --skip-torch; AWS mode never trains the PyTorch model")
    if a.version: ap.error("--version is generated uniquely in AWS mode")
    from tbi_mlops.aws_training import run_aws_training
    run_aws_training(cfg)
    raise SystemExit(0)

out = train_reliability_stack(cfg, version=a.version)
m = out["metrics"]["calibrated"]; r = out["reliability"]
print(f"[advanced] version={out['version']}  test AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f} F1={m['f1']:.3f} Brier={m['brier']:.4f} ECE={m['ece']:.4f}")
print(f"[advanced] conformal coverage={r['conformal']['empirical_coverage_test']:.3f} (target {1-r['conformal']['alpha']:.2f}) avg set size={r['conformal']['avg_set_size']:.2f}; OOD test flag rate={r['ood']['test_flag_rate']:.3f}")
print(f"[advanced] survival C-index={out['survival']['c_index_test']}  monotone={out['survival']['survival_monotone']}")
try:
    import torch  # noqa
    if not a.skip_torch:
        from tbi_mlops.torch_training import train_torch_models
        res = train_torch_models(cfg); print("[torch]", json.dumps(res, default=str)[:500])
except ImportError:
    print("[torch] PyTorch not installed -> skipped (torch code is present but UNTESTED in the build sandbox)")
