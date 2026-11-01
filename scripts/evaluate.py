"""Builds artifacts/reports/EXPERIMENT_REPORT.md from real pipeline outputs, incl. delayed-label performance demo."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import json
import numpy as np
from tbi_mlops.utils.config import load_config, path
from tbi_mlops.data.dataset import prepare, known
from tbi_mlops.serving.predictor import Predictor
from tbi_mlops.preprocessing.features import tabular_features
from tbi_mlops.monitoring.performance import delayed_performance, warning_lead_time
from tbi_mlops.evaluation import reporting
from tbi_mlops.evaluation.classification import curves

cfg = load_config(); P = Predictor(cfg=cfg); D = prepare(cfg, save=False)
test = known(D["parts"]["test"]); T = tabular_features(test["X"], test["valid"]); p = P.model_fn(T); meta = test["meta"]
dp = {f"now={h}h": delayed_performance(meta, p, P.thr["calibrated_threshold"], P.thr["horizon_hours"], h, cfg["monitoring"]) for h in (24, 48, 96, 400)}
lead = warning_lead_time(meta, p, P.thr["calibrated_threshold"], D["patients"])
base = json.load(open(path("artifacts", "reports", "baseline_metrics.json"))); adv = json.load(open(path("artifacts", "reports", "advanced_report.json")))
drift = json.load(open(path("artifacts", "reports", "drift_simulation.json"))) if path("artifacts", "reports", "drift_simulation.json").exists() else {}
torch_report_path = path("artifacts", "reports", "torch_report.json")
torch_report = json.load(open(torch_report_path)) if torch_report_path.exists() else None
reporting.write_json({"delayed_performance": dp, "warning_lead_time": lead}, "delayed_performance.json")
curv = json.load(open(path("artifacts", "reports", "baseline_curves.json"))) if path("artifacts", "reports", "baseline_curves.json").exists() else {}
curv["bootstrap ensemble (calibrated)"] = adv["curves"]["ensemble_calibrated"]
if torch_report:
      curv["PyTorch Transformer + physiological GAT"] = torch_report["test"]["curves"]
reporting.write_json(curv, "model_curves.json")
graph_paths = [reporting.plot_roc_pr(curv)]
dq = json.load(open(path("artifacts", "reports", "data_quality_report.json"))); gen = json.load(open(path("artifacts", "reports", "synthetic_generation_report.json")))
f = lambda x: "n/a" if x is None else f"{x:.3f}"
L = ["# Experiment Report (auto-generated from pipeline outputs)", "", "> **SYNTHETIC DATA.** Results show the pipeline works on a generated mechanism; they say nothing about clinical performance.", "",
     f"Model version: `{adv['version']}`  |  profile: `{cfg['profile']}`  |  dataset: `{dq['dataset_version']}`", "", "## Dataset",
     f"- Patients: {gen['n_patients']}, raw rows: {gen['n_rows_after_injection']}, windows: {dq['n_windows']}, channels: {dq['channels']}, window length: {dq['window_length']}",
     "- Partition stats: " + "; ".join(f"{k}: {v['windows']} windows / {v['patients']} patients / pos-rate {f(v['positive_rate'])}" for k, v in dq["partition_stats"].items()), "",
     "## Data-quality corrections", "```", json.dumps({k: v for k, v in dq["clean_report"]["steps"].items() if k != "outlier_flags_not_removed"}, indent=1), "```",
     f"Injected (generator report): {json.dumps({k: v for k, v in gen.items() if k.endswith('injected') or isinstance(v, dict) and 'injected' in k})}", "",
     "## Baselines vs reliability ensemble (test)", "| model | AUROC | AUPRC | F1 | Brier | p50 latency ms |", "|---|---|---|---|---|---|"]
for n, v in base.items(): L.append(f"| {n} | {f(v['test']['auroc'])} | {f(v['test']['auprc'])} | {f(v['test']['f1'])} | {f(v['test']['brier'])} | {v['latency']['p50_ms']:.2f} |")
m = adv["metrics"]["calibrated"]; L.append(f"| bootstrap ensemble (calibrated) | {f(m['auroc'])} | {f(m['auprc'])} | {f(m['f1'])} | {f(m['brier'])} | {adv['latency']['p50_ms']:.2f} |")
metric_comparison = {n: v["test"] for n, v in base.items()}
metric_comparison["bootstrap ensemble (calibrated)"] = m
if torch_report:
      tm = torch_report["test"]
      metric_comparison["PyTorch Transformer + physiological GAT"] = {"auroc": tm["auroc"], "auprc": tm["auprc"], "f1": None}
      L.append(f"| PyTorch Transformer + physiological GAT multi-task ensemble | {f(tm['auroc'])} | {f(tm['auprc'])} | n/a | n/a | n/a |")
      member_count = sum(key.startswith("member") for key in torch_report)
      L += ["", "## PyTorch multi-task results", f"- AUROC {f(tm['auroc'])}; AUPRC {f(tm['auprc'])}; C-index {f(tm['c_index'])}; mean ensemble standard deviation {f(tm['ensemble_std_mean'])} ({member_count} members).",
              f"- Dataset: `{torch_report['dataset_version']}`; seed: {torch_report['seed']}; code version: `{torch_report['code_version']}`. Metrics are test-partition outputs on synthetic data."]
r = adv["reliability"]
graph_paths.append(reporting.plot_model_metrics(metric_comparison))
raw_probabilities = P.ens.aggregate(T, P.thr["raw_threshold"])["mean"]
graph_paths.append(reporting.plot_calibration(meta.y.values, raw_probabilities, p, cfg["training"]["reliability"]["ece_bins"]))
graph_paths.append(reporting.plot_global_importance(adv["explainability"]["global"]["features"]))
graph_paths.append(reporting.plot_group_importance(adv["explainability"]["global"]["group_importance"]))
L += ["", "## Calibration & uncertainty", f"- Calibration: {r['calibration']['method']}; ECE raw {f(r['calibration']['ece_raw'])} -> calibrated {f(r['calibration']['ece_calibrated'])}; Brier raw {f(r['calibration']['brier_raw'])} -> {f(r['calibration']['brier_calibrated'])}",
      f"- Conformal (alpha={r['conformal']['alpha']}): empirical test coverage {f(r['conformal']['empirical_coverage_test'])}, avg set size {f(r['conformal']['avg_set_size'])}, sets {r['conformal']['set_distribution']}. {r['conformal']['caveat']}",
      f"- OOD: validation flag rate {f(r['ood']['val_flag_rate'])}, test flag rate {f(r['ood']['test_flag_rate'])}", f"- Ensemble: mean std {f(r['ensemble']['mean_std_test'])}, mean agreement {f(r['ensemble']['mean_agreement_test'])}", "",
      "## Survival (discrete-time hazards, censoring-aware)", f"- C-index {f(adv['survival']['c_index_test'])}, monotone survival: {adv['survival']['survival_monotone']}, test events: {adv['survival']['n_events']}",
      "- Limitations: " + "; ".join(adv["survival"]["limitations"]), "", "## Delayed-label performance (only matured predictions scored)", "```", json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('status', 'n_matured', 'n_pending', 'auroc', 'auprc')} for k, v in dp.items()}, indent=1), "```", f"Warning lead time: {lead}", ""]
example = adv["explainability"]["example_window"]
temporal_attr = np.asarray(example.get("temporal_feature_attribution", []), dtype=float)
temporal_diff = np.asarray(example.get("temporal_feature_differences", []), dtype=float)
if temporal_attr.size:
      strongest = np.unravel_index(np.abs(temporal_attr).argmax(), temporal_attr.shape)
      L += ["", "## Temporal-difference explanation (one held-out synthetic window)",
              f"- Feature/time occlusion: hiding {example['feature_names'][strongest[1]]} at relative step {example['relative_steps'][strongest[0]]} changes model risk by {temporal_attr[strongest]:+.4f} (baseline risk {example['baseline_risk']:.4f}).",
              f"- Adjacent-step observed feature differences are stored in preprocessed feature units; {int(np.isfinite(temporal_diff).sum())} feature-step differences are available. These are sensitivity descriptions, not causal effects."]
      valid_steps = np.asarray(example["valid_steps"], dtype=bool)
      graph_paths.append(reporting.plot_temporal(example["step_attribution"], valid_steps, name="temporal_step_attribution.png"))
      graph_paths.append(reporting.plot_temporal_heatmap(temporal_attr, example["feature_names"], example["relative_steps"], "temporal_feature_occlusion.png", "Feature-by-timestep occlusion sensitivity", "risk change"))
      graph_paths.append(reporting.plot_temporal_heatmap(temporal_diff, example["feature_names"], example["relative_steps"], "temporal_feature_differences.png", "Adjacent-step observed feature differences", "scaled value change"))
if drift:
    L += ["## Cohort drift simulation (synthetic scenarios)", "| scenario | drift | #alert feats | OOD rate | mean risk | ms/window |", "|---|---|---|---|---|---|"]
    for k, v in drift.items(): L.append(f"| {k} | {v['drift_overall']} | {v['n_features_alert']} | {v['ood_rate']:.3f} | {v['mean_risk']:.3f} | {v['avg_latency_ms_per_window']:.1f} |")
L += ["", "## Explainability limitations", *[f"- {x}" for x in adv["explainability"]["limitations"]], "", "## Known limitations", "- Synthetic labels derive from the generator; high AUROC mostly reflects that mechanism.",
      "- Dev profile has few positive windows; confidence intervals are wide. PyTorch results are exploratory and are not evidence of clinical performance.", "- Windows from one patient are correlated; conformal coverage guarantees do not strictly apply."]
L += ["", "## Generated graphs", *[f"- [`{pathlib.Path(graph).name}`](../graphs/{pathlib.Path(graph).name})" for graph in graph_paths]]
open(path("artifacts", "reports", "EXPERIMENT_REPORT.md"), "w").write("\n".join(L)); print("[evaluate] wrote artifacts/reports/EXPERIMENT_REPORT.md")
for k, v in dp.items(): print(f"  {k}: {v['status']} matured={v['n_matured']} pending={v['n_pending']}")
