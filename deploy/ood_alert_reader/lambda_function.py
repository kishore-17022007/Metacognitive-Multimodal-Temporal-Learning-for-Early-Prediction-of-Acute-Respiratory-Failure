import json
import os
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")

BUCKET = os.environ["ALERT_BUCKET"]
KEY = "monitoring/live-ood-alert.json"


def lambda_handler(event, context):
    try:
        response = s3.get_object(Bucket=BUCKET, Key=KEY)
        alert = json.loads(response["Body"].read().decode("utf-8"))

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "has_alert": True,
                "alert": alert
            })
        }

    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")

        if error_code in ("NoSuchKey", "404", "NotFound"):
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "has_alert": False,
                    "alert": None
                })
            }

        raise
