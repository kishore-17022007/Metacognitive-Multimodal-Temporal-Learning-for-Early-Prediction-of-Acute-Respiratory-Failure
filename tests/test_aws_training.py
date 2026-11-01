import hashlib
import io
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from tbi_mlops.aws_training import (
    ARTIFACT_FILES,
    ArtifactUploadError,
    AwsTrainingSettings,
    MissingS3InputError,
    _load_s3_inputs,
    _update_manifest,
    _upload_bundle,
    run_aws_training,
)
from scripts.aws_training_preflight import run_preflight
from tbi_mlops.data.schema import OBS_COLUMNS, PATIENT_COLUMNS


class FakeS3Error(Exception):
    def __init__(self, code, status=None):
        super().__init__(code)
        status = status or {"PreconditionFailed": 412, "ConditionalRequestConflict": 409}.get(code, 500)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.conflict_once = False
        self.failures = {}
        self.fail_report_completion = False
        self.put_calls = []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise FakeS3Error("NoSuchKey")
        body = self.objects[Key]
        return {"Body": io.BytesIO(body), "ETag": f'"{hashlib.md5(body).hexdigest()}"'}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **_kwargs):
        content = Body.read() if hasattr(Body, "read") else Body
        self.put_calls.append({"key": Key, "if_none_match": IfNoneMatch, "if_match": IfMatch})
        failures = self.failures.get(Key, [])
        if failures:
            raise failures.pop(0)
        if self.fail_report_completion and Key.endswith(".json") and "/training-reports/" in Key:
            saved = json.loads(content)
            if saved.get("training_report_status") == "COMPLETED":
                raise FakeS3Error("InternalError", 500)
        current = self.objects.get(Key)
        if IfNoneMatch == "*" and current is not None:
            raise FakeS3Error("PreconditionFailed")
        if IfMatch is not None:
            current_etag = f'"{hashlib.md5(current).hexdigest()}"' if current is not None else None
            if self.conflict_once:
                self.conflict_once = False
                concurrent_manifest = json.loads(current)
                concurrent_manifest["versions"]["concurrent"] = {}
                concurrent_manifest["production"] = "vprod"
                self.objects[Key] = json.dumps(concurrent_manifest).encode()
                raise FakeS3Error("PreconditionFailed")
            if current_etag != IfMatch:
                raise FakeS3Error("PreconditionFailed")
        self.objects[Key] = content
        return {"ETag": f'"{hashlib.md5(content).hexdigest()}"'}


class FakePreflightS3:
    def __init__(self):
        self.meta = SimpleNamespace(region_name="ap-southeast-2")
        self.read_keys = []
        self.list_prefixes = []
        self.write_calls = []

    def get_object(self, Bucket, Key, Range=None):
        self.read_keys.append(Key)
        columns = OBS_COLUMNS if Key.endswith("raw_observations.csv") else PATIENT_COLUMNS
        return {"Body": io.BytesIO((",".join(columns) + "\n").encode())}

    def head_bucket(self, Bucket):
        raise AssertionError("preflight should not require unscoped HeadBucket permission")

    def list_objects_v2(self, Bucket, Prefix, MaxKeys):
        self.list_prefixes.append(Prefix)
        return {"KeyCount": 0}

    def put_object(self, **kwargs):
        self.write_calls.append(kwargs)
        raise AssertionError("preflight must not write objects")


class FakeSTS:
    def get_caller_identity(self):
        return {
            "Account": "283912199308",
            "Arn": "arn:aws:sts::283912199308:assumed-role/TrainingRole/session",
            "UserId": "role-session",
        }


def test_settings_require_bucket_and_allow_prefix_overrides():
    with pytest.raises(ValueError, match="TBI_S3_BUCKET"):
        AwsTrainingSettings.from_env({})

    settings = AwsTrainingSettings.from_env({
        "TBI_S3_BUCKET": "research-bucket",
        "TBI_S3_INPUT_PREFIX": "sandbox/data",
        "TBI_S3_MODEL_PREFIX": "sandbox/models",
        "AWS_REGION": "ap-southeast-2",
    })
    assert settings.input_prefix == "sandbox/data"
    assert settings.model_prefix == "sandbox/models"
    assert settings.region == "ap-southeast-2"


def test_read_only_preflight_checks_identity_inputs_and_isolated_prefix():
    s3 = FakePreflightS3()
    result = run_preflight(
        s3_client=s3,
        sts_client=FakeSTS(),
        environ={
            "TBI_S3_BUCKET": "tbi-arf-mlops-mlops-2026",
            "TBI_S3_INPUT_PREFIX": "datasets/synthetic",
            "TBI_S3_MODEL_PREFIX": "models-aws-training-test",
            "AWS_REGION": "ap-southeast-2",
            "TBI_EXPECTED_AWS_ACCOUNT": "283912199308",
        },
    )

    assert result["status"] == "READ_ONLY_PREFLIGHT_PASSED"
    assert result["identity"]["account"] == "283912199308"
    assert set(s3.read_keys) == {
        "datasets/synthetic/raw_observations.csv",
        "datasets/synthetic/patients.csv",
    }
    assert s3.list_prefixes == ["models-aws-training-test/"]
    assert result["output_prefix_is_separate_from_registry"] is True
    assert result["writes_performed"] is False
    assert not s3.write_calls


def test_read_only_preflight_refuses_existing_models_registry_prefix():
    s3 = FakePreflightS3()
    with pytest.raises(ValueError, match="overlaps the existing"):
        run_preflight(
            s3_client=s3,
            sts_client=FakeSTS(),
            environ={
                "TBI_S3_BUCKET": "tbi-arf-mlops-mlops-2026",
                "TBI_S3_MODEL_PREFIX": "models",
                "AWS_REGION": "ap-southeast-2",
            },
        )
    assert not s3.read_keys and not s3.list_prefixes


def test_missing_required_input_fails_without_synthetic_fallback():
    with pytest.raises(MissingS3InputError, match="raw_observations.csv"):
        _load_s3_inputs(FakeS3(), AwsTrainingSettings("bucket", "datasets/synthetic", "models"))


def test_aws_mode_records_missing_input_failure_without_registering_candidate():
    s3 = FakeS3()

    with pytest.raises(MissingS3InputError, match="raw_observations.csv"):
        run_aws_training({}, s3_client=s3, environ={"TBI_S3_BUCKET": "bucket"})

    report_keys = [key for key in s3.objects if key.startswith("models/training-reports/")]
    assert len(report_keys) == 1
    report = json.loads(s3.objects[report_keys[0]])
    assert report["training_status"] == "FAILED"
    assert report["error"]["type"] == "MissingS3InputError"
    assert "raw_observations.csv" in report["error"]["message"]
    assert "models/manifest.json" not in s3.objects


def test_candidate_bundle_upload_is_create_only(tmp_path):
    for filename in ARTIFACT_FILES:
        (tmp_path / filename).write_text(filename)
    s3 = FakeS3()

    checksums = _upload_bundle(s3, "bucket", "models", "vtest", tmp_path)

    assert set(checksums) == set(ARTIFACT_FILES)
    assert set(s3.objects) == {f"models/vtest/{name}" for name in ARTIFACT_FILES}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _upload_bundle(s3, "bucket", "models", "vtest", tmp_path)


def test_artifact_http_409_retries_then_succeeds(tmp_path, monkeypatch):
    for filename in ARTIFACT_FILES:
        (tmp_path / filename).write_text(filename)
    s3 = FakeS3()
    key = "models/vtest/ensemble.joblib"
    s3.failures[key] = [FakeS3Error("ConditionalRequestConflict", 409)]
    monkeypatch.setattr("tbi_mlops.aws_training.time.sleep", lambda _delay: None)

    checksums = _upload_bundle(s3, "bucket", "models", "vtest", tmp_path)

    assert set(checksums) == set(ARTIFACT_FILES)
    assert len([call for call in s3.put_calls if call["key"] == key]) == 2
    assert all(call["if_none_match"] == "*" for call in s3.put_calls)


def test_artifact_http_409_retry_limit_is_bounded(tmp_path, monkeypatch):
    for filename in ARTIFACT_FILES:
        (tmp_path / filename).write_text(filename)
    s3 = FakeS3()
    key = "models/vtest/ensemble.joblib"
    s3.failures[key] = [FakeS3Error("ConditionalRequestConflict", 409) for _ in range(3)]
    monkeypatch.setattr("tbi_mlops.aws_training.time.sleep", lambda _delay: None)

    with pytest.raises(ArtifactUploadError, match="after 3 attempt") as caught:
        _upload_bundle(s3, "bucket", "models", "vtest", tmp_path)

    assert caught.value.status_code == 409
    assert caught.value.failed_file == "ensemble.joblib"
    assert caught.value.uploaded_files == []
    assert len([call for call in s3.put_calls if call["key"] == key]) == 3


def test_artifact_http_412_is_not_retried_or_overwritten(tmp_path, monkeypatch):
    for filename in ARTIFACT_FILES:
        (tmp_path / filename).write_text(filename)
    s3 = FakeS3()
    key = "models/vtest/ensemble.joblib"
    original = b"existing artifact"
    s3.objects[key] = original
    monkeypatch.setattr("tbi_mlops.aws_training.time.sleep", lambda _delay: pytest.fail("412 must not retry"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _upload_bundle(s3, "bucket", "models", "vtest", tmp_path)

    assert s3.objects[key] == original
    assert len([call for call in s3.put_calls if call["key"] == key]) == 1


def test_manifest_merge_preserves_registered_versions_and_never_promotes():
    s3 = FakeS3()
    key = "models/manifest.json"
    s3.objects[key] = json.dumps({
        "versions": {"vold": {"status": "candidate"}},
        "production": "vprod",
    }).encode()

    _update_manifest(s3, "bucket", key, "vnew", {"status": "candidate"})

    manifest = json.loads(s3.objects[key])
    assert set(manifest["versions"]) == {"vold", "vnew"}
    assert manifest["production"] == "vprod"
    assert manifest["versions"]["vnew"]["status"] == "candidate"


def test_manifest_retries_compare_and_swap_conflict_without_losing_versions():
    s3 = FakeS3()
    key = "models/manifest.json"
    s3.objects[key] = json.dumps({"versions": {"vold": {}}, "production": None}).encode()
    s3.conflict_once = True

    _update_manifest(s3, "bucket", key, "vnew", {"status": "candidate"})

    manifest = json.loads(s3.objects[key])
    assert set(manifest["versions"]) == {"vold", "vnew", "concurrent"}
    assert manifest["production"] == "vprod"


def test_manifest_refuses_duplicate_version_without_replacing_manifest():
    s3 = FakeS3()
    key = "models/manifest.json"
    original = json.dumps({"versions": {"vnew": {"status": "candidate"}}, "production": None}).encode()
    s3.objects[key] = original

    with pytest.raises(FileExistsError, match="already registered"):
        _update_manifest(s3, "bucket", key, "vnew", {"status": "candidate"})

    assert s3.objects[key] == original


def test_aws_mode_publishes_candidate_report_without_running_real_training(monkeypatch):
    import tbi_mlops.data.dataset as dataset_module
    import tbi_mlops.training as training_module

    s3 = FakeS3()
    s3.objects["datasets/synthetic/raw_observations.csv"] = (
        b"patient_id,timestamp,hours_since_admission\nP1,2020-01-01T00:00:00,0\n"
    )
    s3.objects["datasets/synthetic/patients.csv"] = (
        b"patient_id,followup_end_h,event_observed,event_time_h,outcome_known\nP1,24,False,,True\n"
    )
    s3.objects["models/manifest.json"] = json.dumps({
        "versions": {"vold": {"status": "candidate"}},
        "production": "vprod",
    }).encode()
    expected_data = {"prepared": True}

    def fake_prepare(cfg, save, raw, pat):
        assert save is False
        assert len(raw) == 1 and len(pat) == 1
        return expected_data

    def fake_train(cfg, D, version, models_root, persist):
        assert D is expected_data
        assert persist is False
        bundle = Path(models_root) / version
        bundle.mkdir()
        for filename in ARTIFACT_FILES:
            (bundle / filename).write_text(filename)
        return {
            "metrics": {
                "raw": {"auroc": 0.81, "auprc": 0.42},
                "calibrated": {"auroc": 0.83, "auprc": 0.45, "f1": 0.6},
            },
            "reliability": {"conformal": {"empirical_coverage_test": 0.9}},
            "survival": {"c_index_test": 0.7},
            "latency": {"p50_ms": 1.2},
            "thresholds": {"calibrated_threshold": 0.5},
        }

    monkeypatch.setattr(dataset_module, "prepare", fake_prepare)
    monkeypatch.setattr(training_module, "train_reliability_stack", fake_train)

    report = run_aws_training(
        {},
        s3_client=s3,
        environ={"TBI_S3_BUCKET": "bucket"},
    )

    version = report["model_version"]
    manifest = json.loads(s3.objects["models/manifest.json"])
    saved_report = json.loads(s3.objects[f"models/training-reports/{report['training_run_id']}.json"])
    assert report["training_status"] == "SUCCEEDED_CANDIDATE"
    assert report["training_report_status"] == "COMPLETED"
    assert report["candidate_state"] == "REGISTERED"
    assert report["training_started_at"] and report["training_ended_at"]
    assert report["synthetic_data_disclaimer"]
    assert report["evaluation"]["metrics"]["calibrated"]["auroc"] == 0.83
    assert set(manifest["versions"][version]["sha256"]) == set(ARTIFACT_FILES)
    assert manifest["versions"][version]["status"] == "candidate"
    assert manifest["production"] == "vprod"
    assert "vold" in manifest["versions"]
    assert saved_report["training_run_id"] == report["training_run_id"]
    assert saved_report["training_status"] == "SUCCEEDED_CANDIDATE"
    assert saved_report["training_report_status"] == "COMPLETED"
    assert saved_report["dataset"]["identifiers"]["patients.csv"]["etag"]


def test_configured_model_prefix_is_used_for_artifacts_manifest_and_report(monkeypatch):
    _stub_training_modules(monkeypatch)
    s3 = _s3_with_training_inputs()
    model_prefix = "models-aws-training-test"

    report = run_aws_training(
        {},
        s3_client=s3,
        environ={
            "TBI_S3_BUCKET": "tbi-arf-mlops-mlops-2026",
            "TBI_S3_INPUT_PREFIX": "datasets/synthetic",
            "TBI_S3_MODEL_PREFIX": model_prefix,
            "AWS_REGION": "ap-southeast-2",
        },
    )

    version = report["model_version"]
    assert report["model_bundle_uri"] == f"s3://tbi-arf-mlops-mlops-2026/{model_prefix}/{version}/"
    assert report["model_manifest_uri"] == f"s3://tbi-arf-mlops-mlops-2026/{model_prefix}/manifest.json"
    assert report["training_report_uri"].startswith(
        f"s3://tbi-arf-mlops-mlops-2026/{model_prefix}/training-reports/"
    )
    assert set(s3.objects) >= {
        f"{model_prefix}/{version}/{filename}" for filename in ARTIFACT_FILES
    }
    assert f"{model_prefix}/manifest.json" in s3.objects
    assert any(key.startswith(f"{model_prefix}/training-reports/") for key in s3.objects)
    assert not any(key == "models/manifest.json" or key.startswith("models/") for key in s3.objects)


def _stub_training_modules(monkeypatch):
    import tbi_mlops.data.dataset as dataset_module
    import tbi_mlops.training as training_module

    expected_data = {"prepared": True}

    def fake_prepare(cfg, save, raw, pat):
        assert save is False
        assert len(raw) == 1 and len(pat) == 1
        return expected_data

    def fake_train(cfg, D, version, models_root, persist):
        assert D is expected_data
        assert persist is False
        bundle = Path(models_root) / version
        bundle.mkdir()
        for filename in ARTIFACT_FILES:
            (bundle / filename).write_text(filename)
        return {
            "metrics": {
                "raw": {"auroc": 0.81, "auprc": 0.42},
                "calibrated": {"auroc": 0.83, "auprc": 0.45, "f1": 0.6},
            },
            "reliability": {"conformal": {"empirical_coverage_test": 0.9}},
            "survival": {"c_index_test": 0.7},
            "latency": {"p50_ms": 1.2},
            "thresholds": {"calibrated_threshold": 0.5},
        }

    monkeypatch.setattr(dataset_module, "prepare", fake_prepare)
    monkeypatch.setattr(training_module, "train_reliability_stack", fake_train)


def _s3_with_training_inputs():
    s3 = FakeS3()
    s3.objects["datasets/synthetic/raw_observations.csv"] = (
        b"patient_id,timestamp,hours_since_admission\nP1,2020-01-01T00:00:00,0\n"
    )
    s3.objects["datasets/synthetic/patients.csv"] = (
        b"patient_id,followup_end_h,event_observed,event_time_h,outcome_known\nP1,24,False,,True\n"
    )
    return s3


def test_report_upload_failure_leaves_registered_candidate_not_completed(monkeypatch):
    _stub_training_modules(monkeypatch)
    s3 = _s3_with_training_inputs()
    s3.fail_report_completion = True

    with pytest.raises(RuntimeError, match="registered, but its training report remains pending"):
        run_aws_training({}, s3_client=s3, environ={"TBI_S3_BUCKET": "bucket"})

    manifest = json.loads(s3.objects["models/manifest.json"])
    version = next(version for version in manifest["versions"] if version != "vold") if "vold" in manifest["versions"] else next(iter(manifest["versions"]))
    report_keys = [key for key in s3.objects if key.startswith("models/training-reports/")]
    saved_report = json.loads(s3.objects[report_keys[0]])
    assert manifest["versions"][version]["status"] == "candidate"
    assert manifest["production"] is None
    assert saved_report["training_status"] == "CANDIDATE_REGISTERED"
    assert saved_report["training_report_status"] == "IN_PROGRESS"
    assert saved_report["training_status"] != "SUCCEEDED_CANDIDATE"


def test_partial_artifact_failure_records_uploaded_files_and_orphan_prefix(monkeypatch):
    _stub_training_modules(monkeypatch)
    s3 = _s3_with_training_inputs()
    original_put = s3.put_object

    def fail_third_artifact(Bucket, Key, Body, **kwargs):
        if Key.endswith("/ood.joblib"):
            raise FakeS3Error("InternalError", 500)
        return original_put(Bucket, Key, Body, **kwargs)

    s3.put_object = fail_third_artifact
    with pytest.raises(ArtifactUploadError, match="ood.joblib"):
        run_aws_training({}, s3_client=s3, environ={"TBI_S3_BUCKET": "bucket"})

    report_keys = [key for key in s3.objects if key.startswith("models/training-reports/")]
    report = json.loads(s3.objects[report_keys[0]])
    assert report["training_status"] == "FAILED"
    assert report["candidate_state"] == "ORPHANED_PARTIAL_UPLOAD"
    assert report["uploaded_artifacts"] == ["ensemble.joblib", "calibrator.joblib"]
    assert report["failed_artifact"]["filename"] == "ood.joblib"
    assert report["orphaned_candidate_prefix_uri"].startswith("s3://bucket/models/v")
    assert "models/manifest.json" not in s3.objects
    assert s3.objects[f"models/{report['model_version']}/ensemble.joblib"] == b"ensemble.joblib"