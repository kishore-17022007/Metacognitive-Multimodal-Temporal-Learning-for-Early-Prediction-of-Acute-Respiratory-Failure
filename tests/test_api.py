import unittest, json, py_compile, glob
from helpers import *
from tbi_mlops.data.schema import *
from tbi_mlops.serving.predictor import validate_records, InputError

REQ = ["prediction_status", "predicted_probability", "prediction_horizon_hours", "uncertainty_summary", "ood_status", "conformal_set", "data_quality_flags", "model_version", "inference_latency_ms"]


def _records(n=10):
    g, _ = long_patient("nonevent"); g = g.sort_values(TIME_COL).head(n).copy(); g[TIME_COL] = g[TIME_COL].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return [{k: (None if (isinstance(v, float) and v != v) else v) for k, v in r.items() if k in [ID_COL, TIME_COL, HOURS_COL] + FEATURE_NAMES} for r in g.to_dict("records")]


def test_valid_prediction_has_required_fields_and_json_serialisable():
    out = tiny()["P"].predict_records(_records()); assert all(k in out for k in REQ) and 0 <= out["predicted_probability"] <= 1 and out["prediction_horizon_hours"] == 6
    assert out["prediction_status"] in ("reliable", "review_required", "out_of_distribution", "insufficient_data"); json.dumps(out); assert "_window" not in out and "NOT" in out["disclaimer"]


def test_invalid_input_rejected_and_bad_values_flagged():
    for bad in ([], [{"timestamp": "2020-01-01T00:00:00"}], [{"patient_id": "a", "timestamp": "garbage"}], [{"patient_id": "a", "timestamp": "2020-01-01T00:00:00", "bogus": 1}],
                [{"patient_id": "a", "timestamp": "2020-01-01T00:00:00"}, {"patient_id": "b", "timestamp": "2020-01-01T01:00:00"}]):
        try: validate_records(bad); raise AssertionError(f"accepted {bad}")
        except InputError: pass
    r = _records(); r[-1]["spo2"] = 150; r[-1]["heart_rate"] = -5; out = tiny()["P"].predict_records(r); assert "invalid_values_masked" in out["data_quality_flags"]


def test_short_history_is_insufficient_data():
    assert tiny()["P"].predict_records(_records(2))["prediction_status"] == "insufficient_data"


def test_fastapi_endpoints_if_installed():
    try: from fastapi.testclient import TestClient
    except ImportError: raise unittest.SkipTest("fastapi not installed (API layer UNTESTED in build sandbox)")
    from tbi_mlops.serving import api; api._P["p"] = tiny()["P"]; c = TestClient(api.app)
    assert c.get("/health").json()["model_loaded"]; r = c.post("/predict", json={"observations": _records()}); assert r.status_code == 200 and all(k in r.json() for k in REQ)
    assert c.post("/predict", json={"observations": []}).status_code == 422 and c.get("/metrics").status_code == 200


def test_dashboard_files_compile_and_streamlit_smoke_if_installed():
    for f in glob.glob(str(ROOT / "dashboard" / "*.py")) + [str(ROOT / "app.py")]: py_compile.compile(f, doraise=True)
    try: from streamlit.testing.v1 import AppTest
    except ImportError: raise unittest.SkipTest("streamlit not installed (dashboard UNTESTED beyond compilation)")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run(); assert not at.exception
