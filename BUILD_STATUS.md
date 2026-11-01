# Build and Verification Status
Verified on 2026-10-24 on macOS ARM64, Python 3.11.9, CPU-only PyTorch. All generated data are synthetic. This is a research prototype, not a diagnostic or validated clinical system.

## Environment
- Created `.venv` with Python 3.11.9 and installed `requirements.txt` successfully.
- Preserved the initially present empty Python 3.15 virtual environment as `.venv-py315` rather than replacing it.
- Dependency adjustment: `mlflow>=2.14,<3`. MLflow 3.17 rejected the file-backed `./mlruns` store; MLflow 2.22.5 supports the documented local workflow. Added `streamlit-autorefresh>=1.0.1` so active streaming simulations rerun automatically.
- No AWS resources were created and no data were uploaded.

## Acceptance Criteria
| Acceptance criterion | Status | Evidence |
|---|---|---|
| Create an isolated venv and install requirements | PASS | `.venv/bin/python -V` -> Python 3.11.9; `.venv/bin/python -m pip install -r requirements.txt` completed successfully |
| Run pytest with optional dependencies enabled | PASS | `.venv/bin/pytest -q` -> **49 passed, 0 skipped, 0 failed**; includes two Torch tests, FastAPI endpoint test, Streamlit AppTest smoke test |
| Generate synthetic data | PASS | `python -m src.tbi_mlops.data.generate_synthetic` -> 22,647 rows, 800 patients |
| Prepare/clean data with leakage safeguards | PASS | `python scripts/prepare_data.py` -> patient-level partitions generated; train-only preprocessing and causal window tests pass |
| Train baseline models | PASS | `python scripts/train_baseline.py` -> LR AUROC 0.897, RF 0.925, HistGB 0.938 on synthetic test partition |
| Train reliability and survival stack | PASS | `python scripts/train_advanced.py` -> calibrated AUROC 0.936, AUPRC 0.731; conformal coverage 0.915; survival C-index 0.934 |
| Train PyTorch multi-task ensemble on CPU | PASS | `python scripts/train_advanced.py` -> 3 members; AUROC 0.934, AUPRC 0.753, C-index 0.923, mean ensemble std 0.008; recorded in `artifacts/reports/torch_report.json` and experiment report |
| PyTorch forward, training step, checkpoint save/load | PASS | `pytest -q tests/test_models.py` -> forward shapes, backward step, `state_dict` save/load with `torch.load(weights_only=True)` all pass |
| Early stopping and survival censoring loss | PASS | CPU ensemble training stops at 13-15 epochs under configured 15-epoch ceiling; Torch test distinguishes event and censoring likelihoods with asymmetric logits |
| GAT shape, neighbor softmax, graph mask | PASS | `test_torch_attention_masks_and_padding_are_respected` -> attention shape, normalization over neighbors and zero attention on non-edges pass |
| Transformer masks padded steps | PASS | Same focused Torch test mutates padded values and verifies invariant output |
| Run simulation and evaluation | PASS | `python simulation/run_simulation.py` -> all seven scenarios and 48 streamed predictions; `python scripts/evaluate.py` -> report written with measured Torch comparison |
| Full local verification, with no optional skips | PASS | `python scripts/verify_local.py` -> 14 checks PASS, 0 FAIL, 0 SKIPPED; includes no-auto-promotion assertion |
| MLflow captures params, metrics, dataset and code versions | PASS | Baseline/reliability/Torch experiments contain runs; newest Torch run has seed 42, dataset `synthetic-2137239112`, code version `unversioned`, AUROC/AUPRC/C-index/ensemble std metrics |
| MLflow UI lists experiments | PASS | `mlflow ui --backend-store-uri ./mlruns`; opened local UI at `http://127.0.0.1:5001`; experiment list shows `tbi-arf-baselines`, `tbi-arf-reliability`, `tbi-arf-torch` |
| API GET endpoints and valid POST endpoints | PASS | Live Uvicorn at `http://127.0.0.1:8765`; GET `/health`, `/model-info`, `/metrics`, `/monitoring/summary` and POST `/predict`, `/predict-window` returned 200 |
| API invalid requests and missing-model behavior | PASS | Invalid/empty POSTs returned 422; with latest artifact directory temporarily moved and restored in `finally`, health returned degraded/200 and model-info/predict returned 503 without traceback response |
| API does not expose patient values in logs/responses | PASS | Requests used marker `privacy-marker-8c241`; marker absent from Uvicorn output and workspace search; responses contain no marker or traceback |
| Streamlit dashboard pages and simulation controls | PASS | All seven pages rendered with AppTest; all seven replay scenarios rendered, and Start/Pause/Resume/Step/Reset completed without exceptions; streaming clock advanced from 0.0 to 0.1 across reruns |
| Graphs for all trained models and temporal-difference explainability | PASS | `artifacts/graphs/` contains saved ROC/PR, metric comparison, calibration, global/group importance, temporal step, occlusion, and difference images; `model_curves.json` contains the five trained model curves. Dashboard renders comparative ROC/PR and both temporal heatmaps |
| Streamlit plots use pipeline outputs and show uncertainty/OOD/conformal/event data | PASS | Streaming page rendered two Plotly charts from real observations/predictions, including ensemble-std band, event annotation, OOD and conformal fields; fixed a pandas column access bug |
| Research/synthetic disclaimers and explicit promotion retained | PASS | README, API response, and dashboard show synthetic/research/not-diagnosis language; registry manifest retains production `null` and candidate status |
| AWS deployment/cloud creation prohibited | PASS | No deploy commands, cloud resources, uploads, or real patient data used; `aws/` remains documentation only |

## Measured Torch Results
Dev profile on synthetic dataset `synthetic-2137239112`: AUROC 0.9336, AUPRC 0.7534, C-index 0.9234, mean ensemble standard deviation 0.0080. These figures do not show consistent superiority to baselines: the PyTorch AUROC is below HistGradientBoosting (0.938), while its AUPRC is above the measured HistGradientBoosting AUPRC (0.712). No clinical conclusions are supported.

## Bugs Found and Fixed
- MLflow 3 rejected the file-backed local tracking backend. Constrained the dependency to MLflow 2.x and verified UI/run logging.
- Existing sklearn pickle artifacts were incompatible with the freshly installed sklearn version and referenced unavailable old module names. Regenerated model artifacts through the normal pipeline in the configured environment.
- Streamlit streaming uncertainty plotting accessed `pr.std`, which resolves to the pandas method rather than the `std` column. Changed to `pr["std"]` and exercised the page with generated predictions.
- API launched as `src.tbi_mlops.serving.api` could not resolve top-level `tbi_mlops` class names embedded in joblib artifacts. Added the source directory to the module search path before artifact loading.
- Torch MLflow runs omitted dataset version. Added dataset version logging and verified it on the newest run.
- Torch test previously gave identical hazard logits for events and censoring, so it could not distinguish those loss paths. Strengthened the assertion with asymmetric logits and explicit expected likelihoods; also added GAT and Transformer masking checks.
- Added the Streamlit auto-refresh dependency required for the running simulation clock to advance automatically.
- Added per-model test ROC/PR overlays and per-feature timestep occlusion plus adjacent-step difference heatmaps; labels explicitly state these are model sensitivity, not causal effects.

## Remaining Limitations
- MLflow retains earlier Torch runs with missing dataset-version params or the previous `unversioned` code marker as historical records; the latest run has complete dataset and source-fingerprint metadata.
- Streamlit produced deprecation warnings for `use_container_width`; the current installed version still renders all pages correctly.
- There are no archived screenshots for the seven pages; pages and controls were checked through AppTest and the running browser UI.
- PyTorch emitted non-failing nested-tensor and tensor-to-float warnings. The tests/training pass; these warnings are not represented as errors.
- Synthetic-label dependence, small dev-profile positive counts, lack of external validation, patient-correlated windows, no IPCW survival Brier, and unvalidated clinical utility remain as described in README.
- AWS deployment intentionally not attempted.

## Commands Run
```bash
python -m src.tbi_mlops.data.generate_synthetic
python scripts/prepare_data.py
python scripts/train_baseline.py
python scripts/train_advanced.py
python simulation/run_simulation.py
python scripts/evaluate.py
python scripts/verify_local.py
python -m pytest -q
uvicorn src.tbi_mlops.serving.api:app --reload
streamlit run app.py
mlflow ui --backend-store-uri ./mlruns
```