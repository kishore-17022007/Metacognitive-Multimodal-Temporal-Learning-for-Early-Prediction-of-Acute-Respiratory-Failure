# AWS (future phase — NOTHING has been deployed)
No cloud resources were created, no data uploaded, no charges incurred. Deployment is a separate, reviewed phase that starts only after local verification (`python scripts/verify_local.py`) passes **and** the torch/FastAPI/Streamlit/MLflow components have been verified on a machine with those libraries.
Artifact access is already behind `serving/storage.py` (`LocalStorage` tested; `S3Storage` stub untested). Credentials: use the AWS provider chain (`AWS_PROFILE`, SSO, instance roles) — never keys in code or `.env` in git.

For the opt-in scikit-learn candidate-training mode, its environment variables, execution command, and least-privilege S3 policy, see [training.md](training.md). This guide does not deploy or promote a model.
