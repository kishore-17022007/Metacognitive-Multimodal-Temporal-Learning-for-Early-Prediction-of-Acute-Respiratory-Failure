import base64
import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


with patch.dict(sys.modules, {"boto3": SimpleNamespace(client=lambda *_args, **_kwargs: None)}):
    inference_handler = importlib.import_module("deploy.lambda_container.handler")


class StubPredictor:
    def __init__(self):
        self.records = None

    def predict_records(self, records):
        self.records = records
        return {"prediction_status": "insufficient_data"}


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("record 0: unknown fields ['private_field']", "unknown_fields"),
        ("record 0: unparseable timestamp PRIVATE-TIMESTAMP", "timestamp_invalid"),
        ("all records must belong to one patient_id PRIVATE-ID", "patient_mismatch"),
        ("no usable observations after cleaning PRIVATE-ID", "empty_after_cleaning"),
        ("unrecognized validation failure PRIVATE-ID", "other_input_error"),
    ],
)
def test_input_error_category_maps_to_safe_reason(message, category):
    assert inference_handler.input_error_category(message) == category


@pytest.mark.parametrize("is_base64_encoded", [False, True])
def test_http_api_v2_post_predict_accepts_json_body(is_base64_encoded):
    payload = {
        "observations": [
            {
                "patient_id": "SYNTH-TEST-001",
                "timestamp": "2026-10-25T11:17:59Z",
                "resp_rate": 20,
                "spo2": 96,
            }
        ]
    }
    body = json.dumps(payload)
    if is_base64_encoded:
        body = base64.b64encode(body.encode("utf-8")).decode("ascii")

    predictor = StubPredictor()
    with patch.object(inference_handler, "load_predictor", return_value=predictor):
        result = inference_handler.lambda_handler(
            {
                "version": "2.0",
                "routeKey": "POST /predict",
                "rawPath": "/predict",
                "requestContext": {"http": {"method": "POST"}},
                "body": body,
                "isBase64Encoded": is_base64_encoded,
            },
            None,
        )

    assert result["statusCode"] == 200
    assert predictor.records == payload["observations"]
    assert json.loads(result["body"])["prediction_status"] == "insufficient_data"


def test_http_api_v2_post_predict_rejects_invalid_base64_safely(capsys):
    result = inference_handler.lambda_handler(
        {
            "version": "2.0",
            "routeKey": "POST /predict",
            "rawPath": "/predict",
            "requestContext": {"http": {"method": "POST"}},
            "body": "not-valid-base64!",
            "isBase64Encoded": True,
        },
        None,
    )

    assert result["statusCode"] == 400
    assert json.loads(result["body"]) == {"error": "Invalid request"}
    assert capsys.readouterr().out == "Invalid request diagnostic: Error\n"


def test_input_error_logs_only_safe_category(capsys):
    class InvalidPredictor:
        def predict_records(self, _records):
            raise inference_handler.InputError(
                "record 0: unknown fields ['private_field']; patient PRIVATE-ID"
            )

    with patch.object(
        inference_handler, "load_predictor", return_value=InvalidPredictor()
    ):
        result = inference_handler.lambda_handler(
            {
                "requestContext": {"http": {"method": "POST"}},
                "body": json.dumps({"observations": [{"patient_id": "SYNTH-TEST"}]}),
            },
            None,
        )

    assert result["statusCode"] == 400
    assert json.loads(result["body"]) == {"error": "Invalid request"}
    assert capsys.readouterr().out == "Invalid request diagnostic: unknown_fields\n"
