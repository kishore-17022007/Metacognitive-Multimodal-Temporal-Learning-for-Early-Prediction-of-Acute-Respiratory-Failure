import json
import os
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone

import boto3

sns = boto3.client("sns")
s3 = boto3.client("s3")


def lambda_handler(event, context):
    if not isinstance(event, dict):
        return {"statusCode": 400, "error": "Expected a JSON object"}

    if isinstance(event.get("observations"), list):
        observations = event["observations"]
    elif isinstance(event.get("observation"), dict):
        observations = [event["observation"]]
    else:
        observations = [event]

    if not observations:
        return {"statusCode": 400, "error": "No observations supplied"}

    if not all(isinstance(item, dict) for item in observations):
        return {"statusCode": 400, "error": "Every observation must be an object"}

    patient_ids = set()

    for item in observations:
        if not item.get("patient_id") or not item.get("timestamp"):
            return {"statusCode": 400, "error": "Missing patient_id or timestamp"}
        if item.get("is_synthetic") is not True:
            return {"statusCode": 400, "error": "Synthetic observations only"}

        try:
            datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            return {"statusCode": 400, "error": "Invalid timestamp"}

        patient_ids.add(item["patient_id"])

    if len(patient_ids) != 1:
        return {
            "statusCode": 400,
            "error": "All observations must belong to one synthetic patient",
        }

    observations.sort(
        key=lambda item: datetime.fromisoformat(
            item["timestamp"].replace("Z", "+00:00")
        )
    )

    url = os.environ["PREDICT_URL"].strip()
    match = re.fullmatch(r"\[.*?\]\((https?://[^)]+)\)", url)
    if match:
        url = match.group(1)
    url = url.rstrip("/")

    payload = json.dumps({"observations": observations}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
            status = response.status
    except urllib.error.HTTPError as exc:
        print(json.dumps({
            "event": "prediction_api_error",
            "http_status": exc.code,
        }))
        raise RuntimeError(f"Prediction API returned HTTP {exc.code}") from None
    except Exception as exc:
        print(json.dumps({
            "event": "prediction_request_failed",
            "error_type": type(exc).__name__,
        }))
        raise RuntimeError(
            f"Prediction request failed ({type(exc).__name__})"
        ) from None

    print(json.dumps({
        "event": "synthetic_temporal_replay_completed",
        "http_status": status,
        "observation_count": len(observations),
        "prediction_status": result.get("prediction_status"),
        "ood_status": result.get("ood_status"),
        "model_version": result.get("model_version"),
        "inference_latency_ms": result.get("inference_latency_ms"),
    }, default=str))

    ood = result.get("ood_status") or {}

    if status == 200 and ood.get("flag") is True:
        alert = {
            "alert_type": "SYNTHETIC_RESEARCH_DEMO_OOD",
            "message": "A synthetic replay triggered the model's OOD safeguard.",
            "patient_id": next(iter(patient_ids)),
            "observation_count": len(observations),
            "model_version": result.get("model_version"),
            "ood_score": ood.get("score"),
            "ood_threshold": ood.get("threshold"),
            "prediction_status": result.get("prediction_status"),
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": (
                "Synthetic-data research demonstration only. "
                "Not a diagnosis or a validated clinical alert."
            ),
        }

        try:
            bucket = os.environ["ALERT_BUCKET"]
            s3.put_object(
                Bucket=bucket,
                Key="monitoring/live-ood-alert.json",
                Body=json.dumps(alert, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            print(json.dumps({
                "event": "synthetic_demo_ood_dashboard_alert_saved",
                "model_version": result.get("model_version"),
            }))
        except Exception as exc:
            print(json.dumps({
                "event": "ood_dashboard_alert_save_failed",
                "error_type": type(exc).__name__,
            }))

        topic_arn = os.environ.get("SNS_TOPIC_ARN")
        if topic_arn:
            try:
                sns.publish(
                    TopicArn=topic_arn,
                    Subject="Synthetic MLOps demo: OOD safeguard triggered",
                    Message=json.dumps(alert, indent=2),
                )
                print(json.dumps({
                    "event": "synthetic_demo_ood_alert_published",
                    "model_version": result.get("model_version"),
                }))
            except Exception as exc:
                print(json.dumps({
                    "event": "sns_alert_publish_failed",
                    "error_type": type(exc).__name__,
                }))
        else:
            print(json.dumps({"event": "sns_topic_arn_missing"}))

    return {"statusCode": status, "result": result}
