import base64
import hashlib
import json
import os
import traceback
from pathlib import Path

import boto3
from tbi_mlops.serving.predictor import InputError

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v20261024-171511")
MODEL_BUCKET = os.environ.get(
    "MODEL_BUCKET", "tbi-arf-mlops-mlops-2026"
)
MODEL_PREFIX = f"models/{MODEL_VERSION}/"
MODEL_ROOT = Path("/tmp/models")
DISCLAIMER = (
    "Research prototype using synthetic data. "
    "Not a diagnosis or a validated clinical service."
)

_predictor = None


def input_error_category(message):
    if "unknown fields" in message:
        return "unknown_fields"
    if "unparseable timestamp" in message:
        return "timestamp_invalid"
    if "all records must belong to one patient_id" in message:
        return "patient_mismatch"
    if "no usable observations after cleaning" in message:
        return "empty_after_cleaning"
    return "other_input_error"


def load_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    manifest = json.loads(
        Path("/var/task/manifest.json").read_text()
    )
    entry = manifest.get("versions", {}).get(MODEL_VERSION)
    if not entry or not entry.get("sha256"):
        raise RuntimeError("Pinned model missing from bundled manifest")

    model_dir = MODEL_ROOT / MODEL_VERSION
    model_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=os.environ.get(
        "AWS_REGION", "ap-southeast-2"
    ))

    for filename, expected_hash in entry["sha256"].items():
        # Restrict manifest paths to simple filenames.
        if Path(filename).name != filename:
            raise RuntimeError("Invalid artifact filename in manifest")

        local_path = model_dir / filename
        if not local_path.exists():
            s3.download_file(
                MODEL_BUCKET, MODEL_PREFIX + filename, str(local_path)
            )

        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        if digest != expected_hash:
            local_path.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum verification failed: {filename}")

    from tbi_mlops.serving.predictor import Predictor

    _predictor = Predictor(
        version=MODEL_VERSION,
        models_dir=str(MODEL_ROOT),
    )
    return _predictor


def lambda_handler(event, context):
    try:
        if event.get("requestContext") and event.get("httpMethod") == "OPTIONS":
            return response(200, {"message": "CORS preflight"})

        method = event.get("requestContext", {}).get(
            "http", {}
        ).get("method", event.get("httpMethod", "POST"))

        path = event.get("rawPath", event.get("path", "/predict"))

        if method == "GET" and path.endswith("/health"):
            return response(200, {
                "status": "ok",
                "prediction_status": "demo_only",
                "model_version": MODEL_VERSION,
                "disclaimer": DISCLAIMER,
            })

        if method != "POST":
            return response(405, {"error": "Method not allowed"})

        raw_body = event.get("body", event)
        if event.get("isBase64Encoded") is True:
            raw_body = base64.b64decode(raw_body, validate=True).decode("utf-8")
        if isinstance(raw_body, str):
            payload = json.loads(raw_body)
        else:
            payload = raw_body

        if not isinstance(payload, dict):
            return response(400, {"error": "Expected a JSON object"})

        observations = payload.get("observations")
        if not isinstance(observations, list) or not observations:
            return response(400, {
                "error": "Provide a non-empty observations array"
            })

        result = load_predictor().predict_records(observations)
        return response(200, result)

    except InputError as exc:
        print(f"Invalid request diagnostic: {input_error_category(str(exc))}")
        return response(400, {"error": "Invalid request"})
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"Invalid request diagnostic: {type(exc).__name__}")
        return response(400, {"error": "Invalid request"})
    except Exception as exc:
        # Log diagnostics for operators; never expose them to callers.
        print(f"Inference failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return response(500, {
            "error": "Inference unavailable",
            "disclaimer": DISCLAIMER,
        })


def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "content-type": "application/json",
            "cache-control": "no-store",
        },
        "body": json.dumps(body, default=str),
    }
