"""Checkpoint-based research evaluation on the existing synthetic patient split."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..data.dataset import prepare
from ..data.schema import FEATURE_NAMES, FEATURES, GROUP_OF, N_F
from ..evaluation.calibration import expected_calibration_error
from ..evaluation.classification import pick_threshold
from ..evaluation.survival import brier_at, concordance_index
from ..models.multitask import MultiTaskModel, survival_from_logits
from ..preprocessing.features import tabular_features
from ..serving.predictor import Predictor
from ..utils.config import load_config, path
from ..utils.reproducibility import code_version, file_sha256


def _metric(value):
    value = float(value)
    return value if np.isfinite(value) else None


def binary_metrics(y, probability, threshold: float, bins: int = 10) -> dict:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not len(y):
        return {"status": "no_valid_targets", "n": 0, "n_positive": 0,
                "threshold": float(threshold), "auroc": None, "auprc": None,
                "brier": None, "ece": None, "precision": None, "recall": None,
                "f1": None, "confusion_matrix": [[0, 0], [0, 0]]}

    predicted = (probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )
    two_classes = len(np.unique(y)) == 2
    return {
        "status": "ok" if two_classes else "single_class_labels",
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "threshold": float(threshold),
        "auroc": _metric(roc_auc_score(y, probability)) if two_classes else None,
        "auprc": _metric(average_precision_score(y, probability)) if two_classes else None,
        "brier": _metric(brier_score_loss(y, probability)),
        "ece": _metric(expected_calibration_error(y, probability, bins)),
        "ece_definition": f"equal-width probability bins on [0,1], bins={bins}",
        "precision": _metric(precision),
        "recall": _metric(recall),
        "f1": _metric(f1),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def _predict(model, X, valid, batch_size=256):
    binary, trajectory, survival = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32)
            mask = torch.as_tensor(valid[start:start + batch_size], dtype=torch.bool)
            output = model(x, mask)
            binary.append(torch.sigmoid(output["binary"]).cpu().numpy())
            trajectory.append(torch.sigmoid(output["traj"]).cpu().numpy())
            survival.append(survival_from_logits(output["haz"]).cpu().numpy())
    return tuple(np.concatenate(values) for values in (binary, trajectory, survival))


def _metrics_for_trajectory(meta, probability, bins, interval_hours):
    intervals = []
    for k in range(probability.shape[1]):
        valid = meta[f"trajvalid_{k}"].to_numpy(dtype=bool)
        intervals.append({
            "horizon_hours": float((k + 1) * interval_hours),
            "target_definition": "synthetic event by interval end; only traj_valid targets",
            "metrics": binary_metrics(
                meta.loc[valid, f"traj_{k}"].to_numpy(dtype=int),
                probability[valid, k], 0.5, bins,
            ),
        })
    return intervals


def _survival_metrics(meta, survival):
    times = meta.surv_time.to_numpy(dtype=int)
    events = meta.surv_event.to_numpy(dtype=int)
    result = {
        "n": int(len(meta)),
        "n_events": int(events.sum()),
        "c_index": _metric(concordance_index(times, events, 1 - survival[:, -1])),
        "c_index_definition": "Harrell concordance on comparable event/censoring pairs; window-level pairs are dependent",
        "brier_by_interval": {},
    }
    for k in range(1, survival.shape[1] + 1):
        result["brier_by_interval"][str(k)] = brier_at(times, events, survival[:, k - 1], k)
    return result


def _event_warning_metrics(meta, probabilities, patients, test_patient_ids, threshold):
    event_patients = patients[
        patients.patient_id.isin(test_patient_ids) & patients.outcome_known
    ].set_index("patient_id")
    predictions = meta.assign(probability=probabilities)
    eligible_events = detected_events = eligible_non_events = false_alert_patients = 0
    false_alert_windows = 0
    lead_times = []

    for patient_id, patient in event_patients.iterrows():
        rows = predictions[predictions.patient_id == patient_id]
        rows = rows[rows.label_known.astype(bool)]
        if bool(patient.event_observed):
            rows = rows[rows.t_index_h < float(patient.event_time_h)]
            if rows.empty:
                continue
            eligible_events += 1
            alerts = rows[rows.probability >= threshold]
            if not alerts.empty:
                detected_events += 1
                first_alert = float(alerts.t_index_h.min())
                lead_times.append(float(patient.event_time_h) - first_alert)
        else:
            if rows.empty:
                continue
            eligible_non_events += 1
            alerts = rows[rows.probability >= threshold]
            false_alert_windows += int(len(alerts))
            false_alert_patients += int(not alerts.empty)

    distribution = None
    if lead_times:
        distribution = {
            "n": len(lead_times),
            "min_hours": _metric(np.min(lead_times)),
            "q25_hours": _metric(np.quantile(lead_times, 0.25)),
            "median_hours": _metric(np.median(lead_times)),
            "mean_hours": _metric(np.mean(lead_times)),
            "q75_hours": _metric(np.quantile(lead_times, 0.75)),
            "max_hours": _metric(np.max(lead_times)),
        }
    return {
        "status": "ok" if eligible_events else "no_eligible_events",
        "rule": "first validation-threshold alert before known synthetic event; negative alerts counted only at label-known prediction times",
        "threshold": float(threshold),
        "eligible_events": int(eligible_events),
        "detected_events": int(detected_events),
        "missed_events": int(eligible_events - detected_events),
        "eligible_non_event_patients": int(eligible_non_events),
        "false_alert_patients": int(false_alert_patients),
        "false_alert_windows": int(false_alert_windows),
        "lead_time_distribution_hours": distribution,
        "event_definition": "synthetic generator event_observed/event_time_h; not a clinical endpoint",
    }


def _stress_inputs(X, valid, seed):
    rng = np.random.default_rng(seed)
    scenarios = {}
    selected = [i for i, feature in enumerate(FEATURES)
                if GROUP_OF[feature.name] in {"respiratory", "cardiovascular"}]

    missing = X.copy()
    observed_selected = np.stack(
        [X[:, :, N_F + feature_index] > 0 for feature_index in selected], axis=2
    )
    remove = (rng.random(observed_selected.shape) < 0.30)
    remove &= valid[:, :, None] & observed_selected
    for j, feature_index in enumerate(selected):
        m = remove[:, :, j]
        missing[:, :, feature_index][m] = 0.0
        missing[:, :, N_F + feature_index][m] = 0.0
        missing[:, :, 2 * N_F + feature_index][m] = 1.0
    scenarios["random_vital_missingness_30pct"] = {
        "X": missing,
        "settings": {"seed": seed, "probability_per_selected_observed_cell": 0.30,
                     "groups": ["respiratory", "cardiovascular"],
                     "value_scaled_to_training_mean": True, "observed_mask_set_false": True},
    }

    noisy = X.copy()
    for feature_index in selected:
        observed = (noisy[:, :, N_F + feature_index] > 0) & valid
        noise = rng.normal(0.0, 0.20, size=observed.shape)
        noisy[:, :, feature_index][observed] += noise[observed]
    scenarios["vital_measurement_noise"] = {
        "X": noisy,
        "settings": {"seed": seed, "gaussian_sd": 0.20,
                     "units": "preprocessed standard-deviation units",
                     "groups": ["respiratory", "cardiovascular"],
                     "observed_values_only": True},
    }

    gap = X.copy()
    gap[:, -1, 3 * N_F + 1] += 0.50
    scenarios["recent_gap_channel_shift"] = {
        "X": gap,
        "settings": {"additive_shift": 0.50,
                     "channel": "preprocessed log-gap channel at prediction step",
                     "units": "preprocessed channel units"},
    }
    return scenarios


def _torch_occlusion_example(models, x, valid):
    valid_steps = np.flatnonzero(valid)
    baseline = float(np.mean([
        _predict(model, x[None], valid[None])[0][0] for model in models
    ]))
    effects = []
    for step in valid_steps:
        for feature_index, feature in enumerate(FEATURES):
            occluded = x.copy()
            occluded[step, feature_index] = 0.0
            occluded[step, N_F + feature_index] = 0.0
            occluded[step, 2 * N_F + feature_index] = 1.0
            score = np.mean([
                _predict(model, occluded[None], valid[None])[0][0]
                for model in models
            ])
            effects.append({
                "relative_step": int(step - len(valid) + 1),
                "feature": feature.name,
                "baseline_minus_occluded_risk": float(baseline - score),
            })
    effects.sort(key=lambda row: abs(row["baseline_minus_occluded_risk"]), reverse=True)
    return {
        "method": "single-window feature-by-time occlusion on the Torch ensemble binary head",
        "baseline_probability": baseline,
        "top_effects": effects[:10],
        "interpretation": "predictive sensitivity only; not causal; one synthetic held-out window",
    }


def run_research_evaluation(seed: int = 20261025) -> dict:
    cfg = load_config()
    data = prepare(cfg, save=False)
    saved_torch = json.loads(path("artifacts", "reports", "torch_report.json").read_text())
    dataset_version = f"synthetic-{data['raw_hash']}"
    if saved_torch.get("dataset_version") != dataset_version:
        raise RuntimeError("Saved Torch report dataset version does not match current synthetic data")

    K = int(data["ds"]["K"])
    models = []
    checkpoint_paths = []
    for member in range(int(saved_torch.get("member_count", 3))):
        model_seed = int(saved_torch["seed"]) + member
        checkpoint = path("artifacts", "models", "torch", f"member_seed{model_seed}.pt")
        model = MultiTaskModel(data["ds"]["X"].shape[2], K, data["ds"]["X"].shape[1], cfg["model"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        models.append(model)
        checkpoint_paths.append(checkpoint)

    predictions = {}
    for split in ("validation", "test"):
        part = data["parts"][split]
        member_outputs = [_predict(model, part["X"], part["valid"]) for model in models]
        predictions[split] = {
            "binary": np.mean([out[0] for out in member_outputs], axis=0),
            "trajectory": np.mean([out[1] for out in member_outputs], axis=0),
            "survival": np.mean([out[2] for out in member_outputs], axis=0),
        }

    validation = data["parts"]["validation"]["meta"]
    validation_known = validation.label_known.to_numpy(dtype=bool)
    validation_y = validation.loc[validation_known, "y"].to_numpy(dtype=int)
    if len(np.unique(validation_y)) == 2:
        threshold = pick_threshold(validation_y, predictions["validation"]["binary"][validation_known])
        threshold_selection = "F1-maximizing threshold on known validation labels only"
    else:
        threshold = 0.5
        threshold_selection = "0.5 fallback: validation labels are single-class or empty"

    bins = int(cfg["training"]["reliability"]["ece_bins"])
    metrics = {"binary_horizon_hours": float(data["ds"]["H"]), "threshold": float(threshold),
               "threshold_selection": threshold_selection, "ece_bins": bins, "splits": {}}
    for split in ("validation", "test"):
        meta = data["parts"][split]["meta"]
        known = meta.label_known.to_numpy(dtype=bool)
        y = meta.loc[known, "y"].to_numpy(dtype=int)
        scores = predictions[split]
        metrics["splits"][split] = {
            "binary": binary_metrics(y, scores["binary"][known], threshold, bins),
            "trajectory": _metrics_for_trajectory(
                meta, scores["trajectory"], bins,
                float(data["ds"]["H"]) / int(data["ds"]["K"]),
            ),
            "survival": _survival_metrics(meta, scores["survival"]),
        }

    patients = data["patients"]
    test_ids = data["splits"]["test"]
    test_meta = data["parts"]["test"]["meta"]
    metrics["early_warning"] = _event_warning_metrics(
        test_meta, predictions["test"]["binary"], patients, test_ids, threshold
    )

    test_part = data["parts"]["test"]
    test_known = test_part["meta"].label_known.to_numpy(dtype=bool)
    y_test = test_part["meta"].loc[test_known, "y"].to_numpy(dtype=int)
    baseline_scores = predictions["test"]["binary"][test_known]
    advanced = json.loads(path("artifacts", "reports", "advanced_report.json").read_text())
    cpu_predictor = Predictor(advanced["version"], cfg=cfg)
    cpu_test_features = tabular_features(test_part["X"], test_part["valid"])
    cpu_test_ood = cpu_predictor.ood.flag(cpu_test_features)[1]
    stress = {}
    for name, case in _stress_inputs(test_part["X"], test_part["valid"], seed).items():
        member_scores = [_predict(model, case["X"], test_part["valid"])[0] for model in models]
        shifted = np.mean(member_scores, axis=0)[test_known]
        shifted_cpu_features = tabular_features(case["X"], test_part["valid"])
        shifted_ood = cpu_predictor.ood.flag(shifted_cpu_features)[1]
        stress[name] = {
            "sample_count": int(len(y_test)),
            "settings": case["settings"],
            "baseline_test_metrics": binary_metrics(y_test, baseline_scores, threshold, bins),
            "shifted_test_metrics": binary_metrics(y_test, shifted, threshold, bins),
            "cpu_serving_ood": {
                "baseline_test_flag_rate": float(cpu_test_ood.mean()),
                "shifted_test_flag_rate": float(shifted_ood.mean()),
                "threshold_source": "pretrained CPU serving artifact; threshold selected on validation data",
            },
            "interpretation": "synthetic input perturbation; original test labels and inputs remain unchanged",
        }
    metrics["stress_tests"] = stress
    metrics["torch_occlusion_example"] = _torch_occlusion_example(
        models, test_part["X"][0], test_part["valid"][0]
    )

    report = {
        "title": "Research Evaluation: TBI Acute Respiratory Failure Prediction",
        "data_provenance": "synthetic",
        "disclaimer": "Research prototype on synthetic data. Not a diagnosis or clinical validation.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "dataset_hash": int(data["raw_hash"]),
        "seed": int(cfg["training"]["seed"]),
        "stress_seed": int(seed),
        "configuration": cfg,
        "evaluation_code_version": code_version(),
        "torch_training_record": {
            "code_version": saved_torch.get("code_version"),
            "seed": saved_torch.get("seed"),
            "checkpoint_sha256": {Path(p).name: file_sha256(p) for p in checkpoint_paths},
            "checkpoint_report_dataset_match": True,
        },
        "split_patient_counts": {name: int(len(ids)) for name, ids in data["splits"].items()},
        "patient_splits_disjoint": all(
            not (data["splits"][a] & data["splits"][b])
            for a, b in (("train", "validation"), ("train", "calibration"),
                         ("train", "test"), ("validation", "calibration"),
                         ("validation", "test"), ("calibration", "test"))
        ),
        "preprocessing_fit_scope": "training patients only (existing prepare pipeline)",
        "offline_model": "PyTorch Transformer + physiological graph multi-task ensemble; existing checkpoints, no retraining",
        "serving_model": "CPU reliability stack loaded by src/tbi_mlops/serving/predictor.py; PyTorch checkpoints are not used by Lambda inference",
        "cpu_reliability_existing_results": {
            "model_version": advanced.get("version"),
            "test_metrics": advanced.get("metrics", {}).get("calibrated"),
            "reliability": advanced.get("reliability"),
            "survival": advanced.get("survival"),
            "validation_window_count": int(len(data["parts"]["validation"]["meta"])),
            "test_window_count": int(len(data["parts"]["test"]["meta"])),
            "ece_definition": f"existing equal-width probability bins on [0,1], bins={cfg['training']['reliability']['ece_bins']}",
            "ood_threshold_selection": "IsolationForest fit on training windows; anomaly cutoff is validation-score quantile, not test-selected",
            "calibration_and_conformal_split": "calibrator and conformal quantile are fit on disjoint patient subsets of the calibration partition",
            "interpretation_warning": "Ensemble member agreement/spread is descriptive and does not establish reliable uncertainty.",
            "conformal_empty_set_count": advanced.get("reliability", {}).get("conformal", {}).get("set_distribution", {}).get("empty"),
            "conformal_empty_set_interpretation": "For LAC scores 1-p_true, an empty set occurs when both class nonconformity scores exceed qhat. This is possible when qhat is small; empty sets are retained and counted, not silently converted to singleton sets.",
            "validation_metric_availability": "The existing advanced report stores CPU-model test metrics but not a complete validation metric block; validation threshold selection and OOD threshold construction are documented in training.py.",
            "ood_evaluation_on_torch_stress_inputs": "The existing CPU OOD detector and validation-selected threshold are evaluated on the same synthetic test windows and deterministic perturbed versions; no OOD model or threshold is refit.",
            "note": "Existing saved CPU-model report; not metrics for the Torch model evaluated here.",
        },
        "metrics": metrics,
        "unavailable_or_limited": [
            "No real clinical validation or real-world distribution-shift evaluation is possible with synthetic-only data.",
            "Survival Brier scores use the existing complete-case censoring simplification without IPCW.",
            "Torch checkpoints match the saved report dataset version, but the old report has no checkpoint hashes; hashes are recorded for this evaluation.",
            "Event-warning results use synthetic event_time_h and validation-selected threshold; they are not clinical lead-time estimates.",
        ],
    }

    json_path = path("artifacts", "reports", "research_evaluation.json")
    markdown_path = path("artifacts", "reports", "RESEARCH_EVALUATION.md")
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    markdown_path.write_text(render_markdown(report))
    return report


def render_markdown(report: dict) -> str:
    metrics = report["metrics"]
    test = metrics["splits"]["test"]["binary"]
    warning = metrics["early_warning"]
    rows = [
        "# Research Evaluation: TBI Acute Respiratory Failure Prediction", "",
        "> **SYNTHETIC DATA ONLY.** This is not clinical validation, a diagnosis, or a validated clinical tool.", "",
        f"- Generated (UTC): {report['generated_at_utc']}",
        f"- Dataset: `{report['dataset_version']}`; seed: {report['seed']}; stress seed: {report['stress_seed']}",
        f"- Patient splits disjoint: {report['patient_splits_disjoint']}; preprocessing fit: {report['preprocessing_fit_scope']}",
        f"- Offline model: {report['offline_model']}", f"- Serving model: {report['serving_model']}", "",
        "## Binary Six-Hour Evaluation", "",
        f"Threshold `{metrics['threshold']:.6f}` selected on validation only ({metrics['threshold_selection']}).",
        f"Test: n={test['n']}, positives={test['n_positive']}, AUROC={test['auroc']}, AUPRC={test['auprc']}, Brier={test['brier']}, ECE={test['ece']}.", "",
        "## Torch Trajectory Head", "", "Metrics are calculated per interval using only `traj_valid` targets; threshold is 0.5.",
        "| Split | Horizon (h) | n | Positives | AUROC | AUPRC | Brier | ECE | F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        for interval in metrics["splits"][split]["trajectory"]:
            m = interval["metrics"]
            rows.append(f"| {split} | {interval['horizon_hours']} | {m['n']} | {m['n_positive']} | {m['auroc']} | {m['auprc']} | {m['brier']} | {m['ece']} | {m['f1']} |")
    rows += ["", "## Synthetic Event Warning", "",
             f"Eligible events: {warning['eligible_events']}; detected: {warning['detected_events']}; missed: {warning['missed_events']}; eligible non-event patients: {warning['eligible_non_event_patients']}; false-alert patients: {warning['false_alert_patients']}; false-alert windows: {warning['false_alert_windows']}.",
             f"Lead-time distribution (hours): `{json.dumps(warning['lead_time_distribution_hours'], sort_keys=True)}`.",
             "These values use generator-defined event times and are not clinical estimates.", "",
             "## Censoring-Aware Head Metrics", "",
             "Per-split Torch survival C-index and Brier scores are in the JSON report. Brier uses complete cases without IPCW.", "",
             "## Robustness Stress Tests", "", "Baseline and perturbed test metrics use the validation-selected threshold; shifts are simulated only.",
             "| Scenario | n | Perturbation settings | AUROC baseline -> shifted | AUPRC baseline -> shifted | Brier baseline -> shifted |", "|---|---:|---|---:|---:|---:|"]
    rows[-1] = "| Scenario | n | Perturbation settings | AUROC baseline -> shifted | AUPRC baseline -> shifted | Brier baseline -> shifted | CPU serving OOD baseline -> shifted |"
    rows[-2] = "|---|---:|---|---:|---:|---:|---:|"
    for name, result in metrics["stress_tests"].items():
        base, shifted = result["baseline_test_metrics"], result["shifted_test_metrics"]
        ood = result["cpu_serving_ood"]
        rows.append(f"| {name} | {result['sample_count']} | `{json.dumps(result['settings'], sort_keys=True)}` | {base['auroc']} -> {shifted['auroc']} | {base['auprc']} -> {shifted['auprc']} | {base['brier']} -> {shifted['brier']} | {ood['baseline_test_flag_rate']:.3f} -> {ood['shifted_test_flag_rate']:.3f} |")
    rows += ["", "## Existing CPU Reliability Report", "",
             f"CPU serving model version: `{report['cpu_reliability_existing_results']['model_version']}`. Its calibration, conformal, OOD, ensemble and survival results are preserved in the JSON report and are distinct from Torch metrics.",
             "The existing CPU report contains empty conformal sets; this is explicitly retained in the saved reliability summary and must not be interpreted as a complete prediction set.", "",
             "## Torch Predictive Occlusion Example", "",
             f"{metrics['torch_occlusion_example']['interpretation']}. Baseline probability: {metrics['torch_occlusion_example']['baseline_probability']:.6f}. Largest effects: `{json.dumps(metrics['torch_occlusion_example']['top_effects'][:5], sort_keys=True)}`.", "",
             "## Limitations", "", *[f"- {item}" for item in report["unavailable_or_limited"]], ""]
    return "\n".join(rows)