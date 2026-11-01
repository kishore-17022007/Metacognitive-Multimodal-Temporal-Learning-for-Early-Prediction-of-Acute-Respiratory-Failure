"""Read-only preflight for the isolated AWS candidate-training configuration."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbi_mlops.aws_training import AwsTrainingSettings
from tbi_mlops.data.schema import OBS_COLUMNS, PATIENT_COLUMNS


REGISTRY_PREFIX = "models"
INPUT_COLUMNS = {
    "raw_observations.csv": set(OBS_COLUMNS),
    "patients.csv": set(PATIENT_COLUMNS),
}


def _client(service: str, region: str):
    import boto3

    return boto3.client(service, region_name=region)


def _read_csv_header(s3, bucket: str, key: str, required_columns: set[str]) -> dict:
    response = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-65535")
    body = response["Body"]
    try:
        header_line = body.readline()
    finally:
        body.close()
    if not header_line:
        raise ValueError(f"S3 CSV is empty: s3://{bucket}/{key}")
    try:
        header = next(csv.reader([header_line.decode("utf-8-sig")]))
    except (UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise ValueError(f"Could not read CSV header: s3://{bucket}/{key}") from error
    missing = sorted(required_columns - set(header))
    if missing:
        raise ValueError(f"CSV is missing required columns at s3://{bucket}/{key}: {missing}")
    return {"uri": f"s3://{bucket}/{key}", "header_columns": len(header), "required_columns_present": True}


def run_preflight(s3_client=None, sts_client=None, environ=None) -> dict:
    env = os.environ if environ is None else environ
    settings = AwsTrainingSettings.from_env(env)
    if not settings.region:
        raise ValueError("Set AWS_REGION or AWS_DEFAULT_REGION before running the preflight")
    if settings.model_prefix == REGISTRY_PREFIX or settings.model_prefix.startswith(REGISTRY_PREFIX + "/"):
        raise ValueError(
            f"Refusing output prefix {settings.model_prefix!r}: it overlaps the existing {REGISTRY_PREFIX!r} registry"
        )
    expected_account = env.get("TBI_EXPECTED_AWS_ACCOUNT")
    s3 = s3_client or _client("s3", settings.region)
    sts = sts_client or _client("sts", settings.region)

    identity = sts.get_caller_identity()
    if expected_account and identity.get("Account") != expected_account:
        raise ValueError(
            f"AWS account mismatch: expected {expected_account}, got {identity.get('Account')}"
        )
    if getattr(s3.meta, "region_name", settings.region) != settings.region:
        raise ValueError(
            f"S3 client region mismatch: expected {settings.region}, got {s3.meta.region_name}"
        )

    inputs = {}
    for filename, required_columns in INPUT_COLUMNS.items():
        key = f"{settings.input_prefix}/{filename}"
        inputs[filename] = _read_csv_header(s3, settings.bucket, key, required_columns)

    output_prefix = settings.model_prefix.rstrip("/") + "/"
    output_listing = s3.list_objects_v2(Bucket=settings.bucket, Prefix=output_prefix, MaxKeys=1)
    return {
        "status": "READ_ONLY_PREFLIGHT_PASSED",
        "identity": {
            "account": identity.get("Account"),
            "arn": identity.get("Arn"),
            "user_id": identity.get("UserId"),
        },
        "region": settings.region,
        "bucket": settings.bucket,
        "inputs": inputs,
        "output_prefix": output_prefix,
        "output_prefix_list_accessible": True,
        "output_prefix_has_objects": bool(output_listing.get("KeyCount", 0)),
        "existing_registry_prefix": REGISTRY_PREFIX + "/",
        "output_prefix_is_separate_from_registry": True,
        "writes_performed": False,
    }


def main() -> int:
    try:
        result = run_preflight()
    except Exception as error:
        print(json.dumps({"status": "PREFLIGHT_FAILED", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())