# Experiment Report (auto-generated from pipeline outputs)

> **SYNTHETIC DATA.** Results show the pipeline works on a generated mechanism; they say nothing about clinical performance.

Model version: `v20261024-171511`  |  profile: `dev`  |  dataset: `synthetic-2137239112`

## Dataset
- Patients: 800, raw rows: 22647, windows: 10086, channels: 47, window length: 12
- Partition stats: train: 5479 windows / 426 patients / pos-rate 0.043; validation: 1507 windows / 116 patients / pos-rate 0.044; calibration: 1495 windows / 115 patients / pos-rate 0.043; test: 1605 windows / 117 patients / pos-rate 0.037

## Data-quality corrections
```
{
 "rows_removed_bad_timestamp": 45,
 "duplicates_removed": 289,
 "conflicting_duplicates_kept_first": 62,
 "unit_corrections": {
  "spo2": 66,
  "fio2": 75,
  "paco2": 22,
  "hemoglobin": 19
 },
 "invalid_set_to_missing": {
  "gcs_total": 19,
  "pupil_response": 0,
  "sedated": 0,
  "resp_rate": 20,
  "spo2": 17,
  "o2_support": 0,
  "fio2": 0,
  "pao2": 0,
  "paco2": 0,
  "heart_rate": 15,
  "sbp": 15,
  "dbp": 0,
  "map": 0,
  "lactate": 0,
  "hemoglobin": 0
 }
}
```
Injected (generator report): {"unit_errors_injected": {"spo2": 68, "fio2": 76, "paco2": 22, "hemoglobin": 19}, "invalid_values_injected": 89, "outliers_injected": 67, "bad_dtype_cells_injected": 1676, "corrupted_cells_injected": 670, "bad_timestamps_injected": 44, "exact_duplicates_injected": 223, "conflicting_duplicates_injected": 67}

## Baselines vs reliability ensemble (test)
| model | AUROC | AUPRC | F1 | Brier | p50 latency ms |
|---|---|---|---|---|---|
| logistic_regression | 0.897 | 0.623 | 0.640 | 0.096 | 0.13 |
| random_forest | 0.925 | 0.703 | 0.710 | 0.027 | 4.10 |
| hist_gradient_boosting | 0.938 | 0.712 | 0.661 | 0.017 | 1.18 |
| bootstrap ensemble (calibrated) | 0.936 | 0.731 | 0.679 | 0.017 | 10.27 |
| PyTorch Transformer + physiological GAT multi-task ensemble | 0.934 | 0.753 | n/a | n/a | n/a |

## PyTorch multi-task results
- AUROC 0.934; AUPRC 0.753; C-index 0.923; mean ensemble standard deviation 0.008 (3 members).
- Dataset: `synthetic-2137239112`; seed: 42; code version: `src-cf304b9b99a3`. Metrics are test-partition outputs on synthetic data.

## Calibration & uncertainty
- Calibration: sigmoid (Platt); ECE raw 0.006 -> calibrated 0.008; Brier raw 0.017 -> 0.017
- Conformal (alpha=0.1): empirical test coverage 0.915, avg set size 0.920, sets {'empty': 121, '{0}': 1399, '{1}': 1, '{0,1}': 0}. marginal coverage assumes exchangeability; windows within a patient are dependent and drift breaks it
- OOD: validation flag rate 0.051, test flag rate 0.061
- Ensemble: mean std 0.013, mean agreement 0.993

## Survival (discrete-time hazards, censoring-aware)
- C-index 0.934, monotone survival: True, test events: 57
- Limitations: Harrell C on windows (patient-dependent samples); Brier without IPCW; trajectory head only available in the PyTorch multi-task model

## Delayed-label performance (only matured predictions scored)
```
{
 "now=24h": {
  "auroc": 0.9953703703703703,
  "auprc": 0.7361111111111112,
  "status": "ok",
  "n_matured": 436,
  "n_pending": 1085
 },
 "now=48h": {
  "auroc": 0.9450202429149798,
  "auprc": 0.7126594577495257,
  "status": "ok",
  "n_matured": 1013,
  "n_pending": 508
 },
 "now=96h": {
  "auroc": 0.9358882178122903,
  "auprc": 0.7306927247406787,
  "status": "ok",
  "n_matured": 1521,
  "n_pending": 0
 },
 "now=400h": {
  "auroc": 0.9358882178122903,
  "auprc": 0.7306927247406787,
  "status": "ok",
  "n_matured": 1521,
  "n_pending": 0
 }
}
```
Warning lead time: {'n_event_patients_warned': 31, 'median_lead_h': 3.0350447425366234}


## Temporal-difference explanation (one held-out synthetic window)
- Feature/time occlusion: hiding o2_support at relative step 0 changes model risk by +0.0559 (baseline risk 0.0698).
- Adjacent-step observed feature differences are stored in preprocessed feature units; 61 feature-step differences are available. These are sensitivity descriptions, not causal effects.
## Cohort drift simulation (synthetic scenarios)
| scenario | drift | #alert feats | OOD rate | mean risk | ms/window |
|---|---|---|---|---|---|
| normal | ok | 0 | 0.023 | 0.051 | 0.1 |
| increasing_instability | alert | 5 | 0.607 | 0.521 | 0.1 |
| missing_data | alert | 10 | 0.992 | 0.039 | 0.1 |
| abnormal_inputs | alert | 6 | 1.000 | 0.091 | 0.1 |
| gradual_drift | alert | 3 | 0.057 | 0.069 | 0.1 |
| abrupt_drift | alert | 3 | 0.093 | 0.102 | 0.1 |
| high_latency | ok | 0 | 0.023 | 0.051 | 330.0 |

## Explainability limitations
- Attribution is not causal
- Attention/occlusion can be unstable across windows
- Synthetic explanations reflect the synthetic generator

## Known limitations
- Synthetic labels derive from the generator; high AUROC mostly reflects that mechanism.
- Dev profile has few positive windows; confidence intervals are wide. PyTorch results are exploratory and are not evidence of clinical performance.
- Windows from one patient are correlated; conformal coverage guarantees do not strictly apply.

## Generated graphs
- [`roc_pr.png`](../graphs/roc_pr.png)
- [`model_metrics.png`](../graphs/model_metrics.png)
- [`calibration.png`](../graphs/calibration.png)
- [`global_feature_importance.png`](../graphs/global_feature_importance.png)
- [`feature_group_importance.png`](../graphs/feature_group_importance.png)
- [`temporal_step_attribution.png`](../graphs/temporal_step_attribution.png)
- [`temporal_feature_occlusion.png`](../graphs/temporal_feature_occlusion.png)
- [`temporal_feature_differences.png`](../graphs/temporal_feature_differences.png)