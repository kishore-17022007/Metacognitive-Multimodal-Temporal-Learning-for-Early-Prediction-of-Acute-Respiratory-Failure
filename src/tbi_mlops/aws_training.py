"""Opt-in AWS orchestration for publishing scikit-learn reliability candidates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import uuid

import pandas as pd


ARTIFACT_FILES = (
    "ensemble.joblib",
    "calibrator.joblib",
    "ood.joblib",
    "survival.joblib",
    "preprocessor.json",
    "thresholds.json",
    "conformal.json",
    "metadata.json",
)
ARTIFACT_UPLOAD_ATTEMPTS = 3
ARTIFACT_RETRY_BASE_SECONDS = 0.05
SYNTHETIC_DISCLAIMER = (
    "Research prototype trained on SYNTHETIC data. Not a validated clinical tool; "
    "predictions are not diagnoses."
)
_LOGGER = logging.getLogger("tbi_mlops.aws_training")
_LOGGER.setLevel(logging.INFO)
if not _LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _LOGGER.addHandler(_handler)


class MissingS3InputError(FileNotFoundError):
    """An expected training CSV is absent from the configured S3 prefix."""


@dataclass(frozen=True)
class AwsTrainingSettings:
    bucket: str
    input_prefix: str
    model_prefix: str
    region: str | None = None

    @classmethod
    def from_env(cls, environ=None) -> "AwsTrainingSettings":
        env = os.environ if environ is None else environ
        bucket = env.get("TBI_S3_BUCKET", "").strip()
        if not bucket or "/" in bucket:
            raise ValueError("TBI_S3_BUCKET must be set to an S3 bucket name")
        return cls(
            bucket=bucket,
            input_prefix=_normalise_prefix(env.get("TBI_S3_INPUT_PREFIX"), "datasets/synthetic"),
            model_prefix=_normalise_prefix(env.get("TBI_S3_MODEL_PREFIX"), "models"),
            region=env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION"),
        )


def _normalise_prefix(value: str | None, default: str) -> str:
    prefix = (value or default).strip("/")
    if not prefix or any(part in (".", "..") for part in prefix.split("/")):
        raise ValueError(f"Invalid S3 prefix: {value!r}")
    return prefix


def _s3_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _error_code(error: Exception) -> str | None:
    response = getattr(error, "response", {})
    return response.get("Error", {}).get("Code") if isinstance(response, dict) else None


def _is_missing_error(error: Exception) -> bool:
    return _error_code(error) in {"404", "NoSuchKey", "NotFound"}


def _conditional_conflict(error: Exception) -> int | None:
    response = getattr(error, "response", {})
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode") if isinstance(response, dict) else None
    code = _error_code(error)
    if status in (409, 412):
        return status
    if code in {"409", "ConditionalRequestConflict"}:
        return 409
    if code in {"412", "PreconditionFailed"}:
        return 412
    return None


class ArtifactUploadError(RuntimeError):
    def __init__(self, message, uploaded_files, failed_file, s3_uri, status_code=None):
        super().__init__(message)
        self.uploaded_files = list(uploaded_files)
        self.failed_file = failed_file
        self.s3_uri = s3_uri
        self.status_code = status_code


class ExistingArtifactError(FileExistsError):
    def __init__(self, message, uploaded_files, failed_file, s3_uri):
        super().__init__(message)
        self.uploaded_files = list(uploaded_files)
        self.failed_file = failed_file
        self.s3_uri = s3_uri


def _log_event(event: str, **fields) -> None:
    _LOGGER.info(json.dumps({"event": event, **fields}, sort_keys=True, default=str))


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _json_bytes(value) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, default=_json_default).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_model_version() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"v{stamp}-{uuid.uuid4().hex[:8]}"


def _load_s3_inputs(s3, settings: AwsTrainingSettings):
    with tempfile.TemporaryDirectory(prefix="tbi-training-input-") as temp_dir:
        paths = {}
        identifiers = {}
        for filename in ("raw_observations.csv", "patients.csv"):
            key = _s3_key(settings.input_prefix, filename)
            uri = f"s3://{settings.bucket}/{key}"
            try:
                response = s3.get_object(Bucket=settings.bucket, Key=key)
            except Exception as error:
                if _is_missing_error(error):
                    raise MissingS3InputError(f"Required AWS training input is missing: {uri}") from error
                code = _error_code(error) or type(error).__name__
                raise RuntimeError(f"Could not read required AWS training input {uri} (S3 error: {code})") from error
            path = Path(temp_dir) / filename
            body = response["Body"]
            try:
                with path.open("wb") as output:
                    while True:
                        chunk = body.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
            finally:
                body.close()
            paths[filename] = path
            identifiers[filename] = {
                "uri": uri,
                "etag": response.get("ETag"),
                "version_id": response.get("VersionId"),
            }
        try:
            raw = pd.read_csv(paths["raw_observations.csv"], dtype=object)
            patients = pd.read_csv(paths["patients.csv"])
        except Exception as error:
            raise ValueError(f"Could not parse required AWS training CSV inputs: {error}") from error
    return raw, patients, identifiers


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upload_bundle(s3, bucket: str, model_prefix: str, version: str, bundle_dir: Path) -> dict:
    missing = [name for name in ARTIFACT_FILES if not (bundle_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Training did not produce the complete inference bundle; missing: {missing}")
    checksums = {name: _sha256(bundle_dir / name) for name in ARTIFACT_FILES}
    uploaded_files = []
    for filename in ARTIFACT_FILES:
        key = _s3_key(model_prefix, version, filename)
        uri = f"s3://{bucket}/{key}"
        for attempt in range(ARTIFACT_UPLOAD_ATTEMPTS):
            try:
                with (bundle_dir / filename).open("rb") as source:
                    s3.put_object(Bucket=bucket, Key=key, Body=source, IfNoneMatch="*")
                uploaded_files.append(filename)
                break
            except Exception as error:
                conflict = _conditional_conflict(error)
                if conflict == 412:
                    raise ExistingArtifactError(
                        f"Refusing to overwrite existing model artifact: {uri}",
                        uploaded_files,
                        filename,
                        uri,
                    ) from error
                if conflict == 409 and attempt + 1 < ARTIFACT_UPLOAD_ATTEMPTS:
                    _log_event("artifact_upload_conflict_retry", attempt=attempt + 1, artifact=uri, http_status=409)
                    time.sleep(ARTIFACT_RETRY_BASE_SECONDS * (2 ** attempt))
                    continue
                raise ArtifactUploadError(
                    f"Failed to upload model artifact {uri} after {attempt + 1} attempt(s)",
                    uploaded_files,
                    filename,
                    uri,
                    status_code=conflict,
                ) from error
    return checksums


def _update_manifest(s3, bucket: str, manifest_key: str, version: str, record: dict, retries: int = 5) -> None:
    """Merge a candidate record using S3 conditional writes to prevent lost updates."""
    for attempt in range(retries):
        try:
            response = s3.get_object(Bucket=bucket, Key=manifest_key)
            manifest = json.loads(response["Body"].read())
            response["Body"].close()
            etag = response.get("ETag")
            if not etag:
                raise RuntimeError("S3 model manifest has no ETag; refusing an unsafe update")
            condition = {"IfMatch": etag}
        except Exception as error:
            if not _is_missing_error(error):
                raise RuntimeError(f"Could not read S3 model manifest s3://{bucket}/{manifest_key}") from error
            manifest = {"versions": {}, "production": None}
            condition = {"IfNoneMatch": "*"}

        if not isinstance(manifest, dict) or not isinstance(manifest.get("versions", {}), dict):
            raise ValueError("S3 model manifest has an invalid structure; refusing to replace it")
        manifest.setdefault("production", None)
        manifest.setdefault("versions", {})
        if version in manifest["versions"]:
            raise FileExistsError(f"Model version {version} is already registered; refusing to overwrite it")
        manifest["versions"][version] = record
        try:
            s3.put_object(
                Bucket=bucket,
                Key=manifest_key,
                Body=_json_bytes(manifest),
                ContentType="application/json",
                **condition,
            )
            return
        except Exception as error:
            if _conditional_conflict(error) not in (409, 412):
                raise RuntimeError(f"Could not conditionally update S3 model manifest s3://{bucket}/{manifest_key}") from error
            _log_event("manifest_update_conflict", attempt=attempt + 1, manifest_key=manifest_key)
    raise RuntimeError(f"S3 model manifest remained contended after {retries} conditional update attempts")


def _upload_report(s3, bucket: str, key: str, report: dict, etag: str | None = None) -> str:
    try:
        response = s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=_json_bytes(report),
            ContentType="application/json",
            **({"IfMatch": etag} if etag else {"IfNoneMatch": "*"}),
        )
    except Exception as error:
        if _conditional_conflict(error) in (409, 412) and etag is None:
            raise FileExistsError(f"Refusing to overwrite training report s3://{bucket}/{key}") from error
        raise RuntimeError(f"Could not save training report s3://{bucket}/{key}") from error
    new_etag = response.get("ETag")
    if not new_etag:
        raise RuntimeError(f"S3 did not return an ETag for training report s3://{bucket}/{key}")
    return new_etag


def _create_s3_client(region: str | None):
    import boto3

    return boto3.client("s3", region_name=region) if region else boto3.client("s3")


def run_aws_training(cfg: dict, s3_client=None, environ=None) -> dict:
    """Run only the existing scikit-learn stack and publish its candidate to S3."""
    from .data.dataset import prepare
    from .training import train_reliability_stack

    settings = AwsTrainingSettings.from_env(environ)
    s3 = s3_client or _create_s3_client(settings.region)
    run_id = uuid.uuid4().hex
    version = _new_model_version()
    started = _utc_now()
    report_key = _s3_key(settings.model_prefix, "training-reports", f"{run_id}.json")
    report = {
        "training_run_id": run_id,
        "model_version": version,
        "training_started_at": started,
        "training_ended_at": None,
        "training_status": "RUNNING",
        "training_report_status": "IN_PROGRESS",
        "candidate_state": "NOT_STARTED",
        "synthetic_data_disclaimer": SYNTHETIC_DISCLAIMER,
        "dataset": {
            "source": "synthetic CSV inputs in S3",
            "bucket": settings.bucket,
            "input_prefix": settings.input_prefix,
            "identifiers": {
                name: {"uri": f"s3://{settings.bucket}/{_s3_key(settings.input_prefix, name)}"}
                for name in ("raw_observations.csv", "patients.csv")
            },
        },
        "model_manifest_uri": f"s3://{settings.bucket}/{_s3_key(settings.model_prefix, 'manifest.json')}",
        "training_report_uri": f"s3://{settings.bucket}/{report_key}",
    }
    _log_event("aws_training_started", run_id=run_id, model_version=version, bucket=settings.bucket)
    report_etag = None
    candidate_registered = False
    model_bundle_uri = f"s3://{settings.bucket}/{_s3_key(settings.model_prefix, version)}/"
    try:
        report_etag = _upload_report(s3, settings.bucket, report_key, report)
        raw, patients, identifiers = _load_s3_inputs(s3, settings)
        report["dataset"]["identifiers"] = identifiers
        report["dataset"].update({"raw_rows": len(raw), "patient_rows": len(patients)})
        prepared = prepare(cfg, save=False, raw=raw, pat=patients)
        with tempfile.TemporaryDirectory(prefix="tbi-training-model-") as model_root:
            result = train_reliability_stack(
                cfg,
                D=prepared,
                version=version,
                models_root=model_root,
                persist=False,
            )
            bundle_dir = Path(model_root) / version
            try:
                checksums = _upload_bundle(s3, settings.bucket, settings.model_prefix, version, bundle_dir)
            except (ArtifactUploadError, ExistingArtifactError) as error:
                report["uploaded_artifacts"] = error.uploaded_files
                report["failed_artifact"] = {
                    "filename": error.failed_file,
                    "uri": error.s3_uri,
                    "http_status": error.status_code if isinstance(error, ArtifactUploadError) else 412,
                    "error": str(error),
                }
                if error.uploaded_files:
                    report["candidate_state"] = "ORPHANED_PARTIAL_UPLOAD"
                    report["orphaned_candidate_prefix_uri"] = model_bundle_uri
                raise
        report["uploaded_artifacts"] = list(ARTIFACT_FILES)
        report["candidate_state"] = "ARTIFACTS_UPLOADED"
        report["training_status"] = "CANDIDATE_ARTIFACTS_UPLOADED"
        report_etag = _upload_report(s3, settings.bucket, report_key, report, etag=report_etag)
        metrics = result["metrics"]
        record = {
            "created": _utc_now(),
            "status": "candidate",
            "sha256": checksums,
            "meta": {
                "training_run_id": run_id,
                "auroc_test": metrics["calibrated"].get("auroc"),
                "auprc_test": metrics["calibrated"].get("auprc"),
                "data_provenance": "synthetic",
                "dataset_source": report["dataset"],
            },
        }
        manifest_key = _s3_key(settings.model_prefix, "manifest.json")
        _update_manifest(s3, settings.bucket, manifest_key, version, record)
        candidate_registered = True
        report.update({
            "training_status": "CANDIDATE_REGISTERED",
            "training_report_status": "IN_PROGRESS",
            "candidate_state": "REGISTERED",
            "model_bundle_uri": model_bundle_uri,
            "evaluation": {
                "metrics": result["metrics"],
                "reliability": result["reliability"],
                "survival": result["survival"],
                "latency": result["latency"],
                "thresholds": result["thresholds"],
            },
        })
        report_etag = _upload_report(s3, settings.bucket, report_key, report, etag=report_etag)
        _log_event("aws_candidate_registered", run_id=run_id, model_version=version, model_bundle_uri=model_bundle_uri)
    except Exception as error:
        report["training_status"] = "CANDIDATE_REGISTERED_REPORT_PENDING" if candidate_registered else "FAILED"
        report["training_report_status"] = "PENDING" if candidate_registered else "FAILED_RECORDED"
        if not candidate_registered and report.get("candidate_state") != "ORPHANED_PARTIAL_UPLOAD":
            report["candidate_state"] = "NOT_REGISTERED"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["training_ended_at"] = _utc_now()
        if isinstance(error, (ArtifactUploadError, ExistingArtifactError)):
            report["uploaded_artifacts"] = error.uploaded_files
            report["failed_artifact"] = {
                "filename": error.failed_file,
                "uri": error.s3_uri,
                "http_status": error.status_code if isinstance(error, ArtifactUploadError) else 412,
                "error": str(error),
            }
            if error.uploaded_files:
                report["candidate_state"] = "ORPHANED_PARTIAL_UPLOAD"
                report["orphaned_candidate_prefix_uri"] = model_bundle_uri
        if report_etag:
            try:
                report_etag = _upload_report(s3, settings.bucket, report_key, report, etag=report_etag)
            except Exception:
                _LOGGER.exception("Could not persist AWS training failure state for run_id=%s", run_id)
        _LOGGER.exception("AWS training failed for run_id=%s model_version=%s", run_id, version)
        raise

    report["training_status"] = "SUCCEEDED_CANDIDATE"
    report["training_report_status"] = "COMPLETED"
    report["training_ended_at"] = _utc_now()
    try:
        _upload_report(s3, settings.bucket, report_key, report, etag=report_etag)
    except Exception as error:
        _LOGGER.exception("Candidate is registered but its report is not finalized for run_id=%s", run_id)
        raise RuntimeError(
            f"Candidate {version} is registered, but its training report remains pending at "
            f"s3://{settings.bucket}/{report_key}"
        ) from error
    _log_event("aws_training_completed", run_id=run_id, model_version=version, status=report["training_status"])
    return report