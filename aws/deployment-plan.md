# Proposed AWS deployment plan (not executed)
| Concern | Service | Notes |
|---|---|---|
| Artifacts (models, preprocessor, reports) | S3 (versioned bucket, SSE, block public access) | Only approved artifacts. Synthetic data only unless governance approves otherwise. Keep `manifest.json` + SHA-256 verification. |
| Inference | Lambda + API Gateway **if** the package fits (sklearn+numpy+pandas exceed 250 MB unzipped limits quickly; use a container image Lambda or ECS/Fargate otherwise) | Current predictor is CPU-light (~40 ms/window) but loads pickled joblib models: load only from trusted, checksum-verified buckets. |
| Logs/metrics/alerts | CloudWatch Logs, custom metrics (latency, error rate, status distribution), alarms | Logs already avoid patient values. |
| Access | IAM least privilege: inference role = `s3:GetObject` on one prefix + CloudWatch put | No wildcard resources; separate deploy and runtime roles. |
| Cost control | AWS Budgets + billing alarms before any resource is created | Services are not all free; estimate Lambda/API Gateway/S3/CloudWatch costs first. |
| Dashboard | Container (ECS/App Runner) behind auth, or keep local | Streamlit is not designed for public unauthenticated exposure. |
## Pre-deployment checklist
1. All components verified locally (incl. torch, FastAPI TestClient, Streamlit AppTest, MLflow).
2. Security review of pickle usage (consider ONNX / skops) and dependency pinning.
3. Governance review: real patient data requires IRB/DUA, de-identification and a separate environment.
4. Budget + alarms created. 5. Infrastructure as code reviewed (Terraform/CDK) before apply.
