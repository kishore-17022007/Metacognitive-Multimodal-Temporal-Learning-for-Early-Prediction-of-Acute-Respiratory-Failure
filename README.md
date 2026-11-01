# Metacognitive Multimodal Temporal Learning for Early ARF Prediction in TBI
**Research & MLOps prototype on SYNTHETIC data.** Not a validated clinical tool; outputs are risk *estimates*, never diagnoses. The synthetic outcome rule is a documented generator, **not** a clinical criterion.

## Status (verified 2026-10-24)
| Component | Status |
|---|---|
| Synthetic ICU generator with controlled defects, cleaning, before/after quality report | **Executed & verified** |
| Leakage-safe preprocessing, causal imputation, masked windows, patient-level splits | **Executed & verified** |
| Baselines (LR, RF, HistGB; XGBoost if installed) | **Executed & verified** |
| Reliability layer: patient-bootstrap ensemble, calibration, split-conformal, Isolation-Forest OOD, rule-based reliability status | **Executed & verified** |
| Discrete-time survival with censoring (person-period hazards), C-index, Brier | **Executed & verified** |
| Explainability (permutation importance, temporal/group occlusion) | **Executed & verified** (SHAP used only if installed) |
| Per-model ROC/precision-recall plots and feature-by-time temporal sensitivity/difference heatmaps | **Generated from held-out synthetic predictions and rendered in dashboard**; temporal sensitivity is occlusion-based, descriptive, and not causal |
| Streaming engine, all seven scenarios, drift/latency monitoring, delayed-label evaluation | **Executed & verified** |
| Model registry (immutable versions, checksums, explicit promotion), local experiment tracker | **Executed & verified**; no model auto-promoted |
| PyTorch Transformer, physiological GAT, gated-fusion multi-task model, torch survival loss, training/ensemble | **Executed on CPU; tests pass** (see measured synthetic metrics below) |
| FastAPI service and Streamlit dashboard (7 pages) | **Executed & verified**; endpoints and all pages exercised |
| MLflow tracking/UI | **Executed & verified**; MLflow 2.x file store, runs list in UI |
| AWS | Plan only; nothing deployed |

All results use synthetic data and are research-only, not a diagnosis or clinical validation. Dev-profile PyTorch test results: AUROC 0.934, AUPRC 0.753, C-index 0.923, mean ensemble standard deviation 0.008. These do not establish superiority to baselines; see [`artifacts/reports/EXPERIMENT_REPORT.md`](artifacts/reports/EXPERIMENT_REPORT.md).

Generated static graphs are saved under [`artifacts/graphs/`](artifacts/graphs/), including all-model ROC/PR curves, metric comparison, calibration, global/group importance, and temporal attribution/difference heatmaps. Regenerate them with `python scripts/evaluate.py` after training; interactive versions are available on the dashboard.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # installs torch/mlflow/fastapi/streamlit too
make all            # or: python scripts/verify_local.py  (full pipeline + assertions)
```
Step by step: `python -m src.tbi_mlops.data.generate_synthetic` → `python scripts/prepare_data.py` → `python scripts/train_baseline.py` → `python scripts/train_advanced.py` (also trains torch models if torch is present) → `python simulation/run_simulation.py` → `python scripts/evaluate.py` → `python scripts/run_tests.py` (or `pytest -q`).
Serve: `uvicorn src.tbi_mlops.serving.api:app --reload` (docs at `/docs`) · `streamlit run app.py` · `mlflow ui --backend-store-uri ./mlruns`. MLflow is constrained to `<3` because MLflow 3 rejects the file-backed store used by this local workflow.
Profiles: `TBI_PROFILE=dev|research` or `--profile`. Dev runs the full pipeline in ~1 min on 1 CPU.

## Methodology
- **Task:** at each prediction index (every 2nd observation) predict whether a synthetic ARF event occurs within the next **6 h**. Windows = last 12 observations, left-padded + masked; channels = scaled value, observed-flag, observation-age per feature + time-since-admission + inter-observation gap.
- **Labels/censoring:** windows after the event are not generated; negatives require follow-up ≥ horizon; censored-before-horizon windows get **no binary label** but contribute censored survival intervals; unknown-outcome patients are excluded.
- **Leakage controls:** patient-level split before fitting; preprocessor fitted on train patients only; causal forward-fill (tested: altering future rows doesn't change earlier windows; incremental == batch); no outcome-derived fields (tested).
- **Reliability:** ensemble std + member agreement; calibrator fit on half the calibration patients, conformal on the other half; OOD threshold = validation-score quantile; status ∈ `reliable | review_required | out_of_distribution | insufficient_data` via transparent rules in `configs/training.yaml`. The "experimental_score" is an unvalidated heuristic.
- **Conformal caveat:** coverage is marginal and assumes exchangeability; windows within a patient and drift violate it.
- **Monitoring:** cohort PSI/KS/missingness drift; per-patient stream-shift monitor; label-matured performance only; latency/error/status counters; alerts advisory — **no auto-retrain/promotion**.

## Layout
`src/tbi_mlops/{data,preprocessing,models,reliability,explainability,evaluation,monitoring,streaming,serving,tracking,utils}`, `dashboard/`, `scripts/`, `simulation/`, `tests/`, `configs/`, `aws/`. Deviations from the spec tree: added `tracking/`, `training.py`, `torch_training.py`, `models/survival.py` is the numpy discrete-time model (torch survival loss lives in `models/multitask.py`).

## Real data (MIMIC-IV etc.)
Requires PhysioNet credentialing + signed DUA; this repo never downloads data. Use `data/load_data.RealDataAdapter` with a mapping YAML; unmapped features stay NaN (not fabricated). Real labels must be clinician-adjudicated; keep reports separate from synthetic ones (`data_source`/`is_synthetic` columns).

## Known limitations
Synthetic labels follow the generator, so high AUROC mostly reflects that mechanism. Dev profile has ~240 train positives (wide uncertainty). No external validation. Survival Brier lacks IPCW; C-index treats windows as independent. Trajectory head exists only in the torch model. Joblib artifacts are pickles — load only trusted, checksum-verified files. Explanations are not causal and can be unstable.
# Metacognitive-Multimodal-Temporal-Learning-for-Early-Prediction-of-Acute-Respiratory-Failure
