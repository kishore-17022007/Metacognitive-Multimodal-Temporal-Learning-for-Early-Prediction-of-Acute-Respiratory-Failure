# Research Evaluation: TBI Acute Respiratory Failure Prediction

> **SYNTHETIC DATA ONLY.** This is not clinical validation, a diagnosis, or a validated clinical tool.

- Generated (UTC): 2026-10-25T14:29:56.859614+00:00
- Dataset: `synthetic-2137239112`; seed: 42; stress seed: 20261025
- Patient splits disjoint: True; preprocessing fit: training patients only (existing prepare pipeline)
- Offline model: PyTorch Transformer + physiological graph multi-task ensemble; existing checkpoints, no retraining
- Serving model: CPU reliability stack loaded by src/tbi_mlops/serving/predictor.py; PyTorch checkpoints are not used by Lambda inference

## Binary Six-Hour Evaluation

Threshold `0.526412` selected on validation only (F1-maximizing threshold on known validation labels only).
Test: n=1521, positives=57, AUROC=0.9335634167385677, AUPRC=0.7534110917854201, Brier=0.016301833777238206, ECE=0.011279296471610578.

## Torch Trajectory Head

Metrics are calculated per interval using only `traj_valid` targets; threshold is 0.5.
| Split | Horizon (h) | n | Positives | AUROC | AUPRC | Brier | ECE | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 1.0 | 1507 | 11 | 0.8497204666990763 | 0.30167846063129233 | 0.006145626458307869 | 0.0020715021117335133 | 0.0 |
| validation | 2.0 | 1494 | 21 | 0.8737270875763747 | 0.4776538754227344 | 0.009701711094191434 | 0.004096117728406006 | 0.24 |
| validation | 3.0 | 1478 | 27 | 0.8038134619802435 | 0.4638868195749345 | 0.012028581864743787 | 0.005741611071887486 | 0.34146341463414637 |
| validation | 4.0 | 1465 | 39 | 0.8386017909159563 | 0.5547063673839762 | 0.015099741790719257 | 0.006154845508908911 | 0.5846153846153846 |
| validation | 5.0 | 1445 | 48 | 0.8520042949176808 | 0.6651398171713203 | 0.016917922778615674 | 0.007095814412169699 | 0.6153846153846154 |
| validation | 6.0 | 1425 | 62 | 0.8510046623908363 | 0.6591873774285366 | 0.022201733514868986 | 0.008345643292393602 | 0.6846846846846847 |
| test | 1.0 | 1605 | 10 | 0.9910971786833855 | 0.3164774436090225 | 0.004898502952639942 | 0.002932582938056469 | 0.0 |
| test | 2.0 | 1589 | 26 | 0.91397214429844 | 0.49812041927817063 | 0.010471285841393382 | 0.009230324434004537 | 0.29411764705882354 |
| test | 3.0 | 1565 | 28 | 0.9237847383585834 | 0.6042929028846116 | 0.009124360506826504 | 0.010424763674642474 | 0.5652173913043478 |
| test | 4.0 | 1550 | 36 | 0.9322801996183767 | 0.8124027211765911 | 0.009286903068249298 | 0.009504338009808695 | 0.71875 |
| test | 5.0 | 1532 | 45 | 0.93001569154898 | 0.7664245425829775 | 0.012994663832826554 | 0.013353179336596018 | 0.7142857142857143 |
| test | 6.0 | 1521 | 57 | 0.9310348959831272 | 0.7580411645275872 | 0.01624936547771202 | 0.009974981936914535 | 0.7169811320754716 |

## Synthetic Event Warning

Eligible events: 46; detected: 33; missed: 13; eligible non-event patients: 71; false-alert patients: 3; false-alert windows: 4.
Lead-time distribution (hours): `{"max_hours": 11.682982800572425, "mean_hours": 3.6129209411894947, "median_hours": 3.103197635826543, "min_hours": 0.19050452565714338, "n": 33, "q25_hours": 1.2341499662069424, "q75_hours": 5.818448006424298}`.
These values use generator-defined event times and are not clinical estimates.

## Censoring-Aware Head Metrics

Per-split Torch survival C-index and Brier scores are in the JSON report. Brier uses complete cases without IPCW.

## Robustness Stress Tests

Baseline and perturbed test metrics use the validation-selected threshold; shifts are simulated only.
|---|---:|---|---:|---:|---:|---:|
| Scenario | n | Perturbation settings | AUROC baseline -> shifted | AUPRC baseline -> shifted | Brier baseline -> shifted | CPU serving OOD baseline -> shifted |
| random_vital_missingness_30pct | 1521 | `{"groups": ["respiratory", "cardiovascular"], "observed_mask_set_false": true, "probability_per_selected_observed_cell": 0.3, "seed": 20261025, "value_scaled_to_training_mean": true}` | 0.9335634167385677 -> 0.9054740676828683 | 0.7534110917854201 -> 0.574245617431968 | 0.016301833777238206 -> 0.02119778735474763 | 0.058 -> 0.460 |
| vital_measurement_noise | 1521 | `{"gaussian_sd": 0.2, "groups": ["respiratory", "cardiovascular"], "observed_values_only": true, "seed": 20261025, "units": "preprocessed standard-deviation units"}` | 0.9335634167385677 -> 0.9332158949285784 | 0.7534110917854201 -> 0.7582833308880984 | 0.016301833777238206 -> 0.01603632367726251 | 0.058 -> 0.059 |
| recent_gap_channel_shift | 1521 | `{"additive_shift": 0.5, "channel": "preprocessed log-gap channel at prediction step", "units": "preprocessed channel units"}` | 0.9335634167385677 -> 0.9335154826958105 | 0.7534110917854201 -> 0.7522913011310854 | 0.016301833777238206 -> 0.016369819070428525 | 0.058 -> 0.068 |

## Existing CPU Reliability Report

CPU serving model version: `v20261024-171511`. Its calibration, conformal, OOD, ensemble and survival results are preserved in the JSON report and are distinct from Torch metrics.
The existing CPU report contains empty conformal sets; this is explicitly retained in the saved reliability summary and must not be interpreted as a complete prediction set.

## Torch Predictive Occlusion Example

predictive sensitivity only; not causal; one synthetic held-out window. Baseline probability: 0.006281. Largest effects: `[{"baseline_minus_occluded_risk": -0.0014190706424415112, "feature": "fio2", "relative_step": 0}, {"baseline_minus_occluded_risk": 0.0009677414782345295, "feature": "map", "relative_step": 0}, {"baseline_minus_occluded_risk": 0.0007734796963632107, "feature": "sbp", "relative_step": 0}, {"baseline_minus_occluded_risk": 0.0006153900176286697, "feature": "gcs_total", "relative_step": 0}, {"baseline_minus_occluded_risk": -0.0005977950058877468, "feature": "heart_rate", "relative_step": 0}]`.

## Limitations

- No real clinical validation or real-world distribution-shift evaluation is possible with synthetic-only data.
- Survival Brier scores use the existing complete-case censoring simplification without IPCW.
- Torch checkpoints match the saved report dataset version, but the old report has no checkpoint hashes; hashes are recorded for this evaluation.
- Event-warning results use synthetic event_time_h and validation-selected threshold; they are not clinical lead-time estimates.
