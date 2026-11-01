# TBI ARF MLOps — AWS Deployment Plan

Status: Local planning only. No AWS resources have been created.

## Current environment
- AWS CLI is installed.
- Configured region: ap-southeast-2 (Sydney).
- AWS identity check succeeded.
- No S3 buckets were listed.

## Phase 1 — Local validation
1. Verify the project's Python environment.
2. Run the existing tests.
3. Check Python, NumPy, SciPy, scikit-learn and joblib versions.
4. Verify model artifact checksums.
5. Confirm the API loads an explicitly selected model version.
6. Use synthetic data only.

## Phase 2 — Private S3 storage
Potential uses:
- Synthetic datasets
- Versioned model artifacts
- Evaluation reports

Before creating a bucket, confirm Free plan eligibility and expected costs.
Enable Block Public Access, default encryption and appropriate retention.
No bucket has been created.

## Phase 3 — Inference API
Evaluate Lambda and API Gateway only after validating:
- Runtime and dependency compatibility
- Model artifact size and packaging
- Inference latency and memory requirements
- Explicit model-version selection
- Authentication, throttling and logging

Keep model training, Streamlit and MLflow local initially.

## Safety
- Do not upgrade the AWS account plan.
- Do not upload real patient data.
- Review and approve each cloud resource before creation.
- Predictions are research estimates, not clinical diagnoses.
