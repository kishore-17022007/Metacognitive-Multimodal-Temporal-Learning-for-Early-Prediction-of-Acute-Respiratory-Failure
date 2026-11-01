from helpers import *
import os, tempfile, json
from tbi_mlops.reliability.calibration import Calibrator
from tbi_mlops.reliability.conformal import ConformalClassifier
from tbi_mlops.reliability.ood import OODDetector
from tbi_mlops.reliability.assess import decide
from tbi_mlops.reliability.ensemble import Ensemble
from tbi_mlops.evaluation.calibration import expected_calibration_error
from tbi_mlops.tracking import registry
from tbi_mlops.serving.predictor import Predictor, ModelNotAvailable
RC = dict(max_ensemble_std=0.15, min_agreement=0.8, review_probability_band=[0.3, 0.7])


def test_ensemble_aggregation():
    rng = np.random.default_rng(0); X = rng.normal(size=(300, 5)); y = (X[:, 0] + rng.normal(0, .5, 300) > 0).astype(int); ids = np.repeat(np.arange(60), 5)
    e = Ensemble(n_members=3, max_iter=20).fit(X, y, ids); A = e.aggregate(X[:10], 0.5)
    assert A["members"].shape == (10, 3) and np.allclose(A["mean"], A["members"].mean(1)) and np.allclose(A["std"], A["members"].std(1)) and ((A["agreement"] >= 0.5) & (A["agreement"] <= 1)).all()


def test_calibration_improves_and_identity_fallback():
    rng = np.random.default_rng(1); true = rng.uniform(0, 1, 4000); y = (rng.uniform(size=4000) < true).astype(int); over = np.clip(true ** 0.4, 0, 1)   # miscalibrated scores
    c = Calibrator("sigmoid").fit(over[:2000], y[:2000]); assert expected_calibration_error(y[2000:], c.predict(over[2000:])) < expected_calibration_error(y[2000:], over[2000:])
    assert Calibrator().fit(np.array([.1, .2, .3]), np.array([0, 0, 0])).status.startswith("identity")


def test_conformal_coverage_exchangeable():
    rng = np.random.default_rng(2); p = rng.uniform(size=6000); y = (rng.uniform(size=6000) < p).astype(int)
    cc = ConformalClassifier(0.1).fit(p[:3000], y[:3000]); cov, size = cc.coverage(p[3000:], y[3000:]); assert cov >= 0.88, cov
    assert cc.predict_sets([0.5]) is not None and all(set(s) <= {0, 1} for s in cc.predict_sets(p[:50])); assert ConformalClassifier.from_dict(cc.to_dict()).qhat == cc.qhat


def test_ood_threshold_from_validation_and_flags_shift():
    rng = np.random.default_rng(3); tr, va = rng.normal(size=(2000, 47)), rng.normal(size=(600, 47))
    import tbi_mlops.reliability.ood as o; names, groups = o.tabular_feature_names(); k = len(names); tr, va = rng.normal(size=(2000, k)), rng.normal(size=(600, k))
    d = OODDetector(0.05).fit(tr, va); assert abs(d.flag(va)[1].mean() - 0.05) < 0.01
    far = rng.normal(8, 1, size=(100, k)); assert d.flag(far)[1].mean() > 0.95; ex = d.explain(far[0]); assert ex["top_features"] and ex["group_mean_abs_z"]


def test_explicit_unreliable_states():
    ok = decide(0.05, 0.05, 0.01, 1.0, "sigmoid", 0.1, False, [0], [], RC, "v")
    assert ok.status == "reliable" and not ok.reasons
    assert decide(0.05, 0.05, 0.01, 1.0, "s", 0.9, True, [0], [], RC, "v").status == "out_of_distribution"
    assert decide(0.5, 0.5, 0.2, 0.6, "s", 0.1, False, [0, 1], [], RC, "v").status == "review_required"
    assert decide(0.05, 0.05, 0.01, 1.0, "s", 0.1, False, [0], ["insufficient_history"], RC, "v").status == "insufficient_data"
    assert 0 <= ok.experimental_score <= 1 and ok.model_version == "v" and ok.prediction_time


def test_pipeline_reliability_outputs_present():
    r = tiny()["out"]["reliability"]; assert 0.8 <= r["conformal"]["empirical_coverage_test"] <= 1.0
    assert 0 <= r["ood"]["test_flag_rate"] <= 1 and r["calibration"]["method"] and r["ensemble"]["n_members"] == 2


def test_calibrator_artifact_roundtrip_and_predictor_load():
    t = tiny(); P2 = Predictor("vtest", cfg=t["cfg"], models_dir=t["models_dir"]); x = np.linspace(0.01, 0.99, 7)
    assert np.allclose(P2.cal.predict(x), t["P"].cal.predict(x))


def test_missing_model_artifact_behaviour():
    try: Predictor("does-not-exist", models_dir=tempfile.mkdtemp()); raise AssertionError("should fail")
    except ModelNotAvailable: pass


def test_registry_never_overwrites_and_verifies_checksums():
    old = (registry.MODELS, registry.MANIFEST); tmp = tempfile.mkdtemp(); registry.MODELS = __import__("pathlib").Path(tmp); registry.MANIFEST = registry.MODELS / "manifest.json"
    try:
        (registry.MODELS / "v1").mkdir(); (registry.MODELS / "v1" / "f.txt").write_text("a"); registry.register("v1", ["f.txt"], {})
        try: registry.register("v1", ["f.txt"], {}); raise AssertionError("overwrite allowed")
        except FileExistsError: pass
        assert registry.manifest()["production"] is None and registry.verify("v1")
        try: registry.promote("v1", ""); raise AssertionError("promotion without reviewer")
        except ValueError: pass
        registry.promote("v1", "dr.x"); assert registry.manifest()["production"] == "v1"
        (registry.MODELS / "v1" / "f.txt").write_text("tampered"); assert not registry.verify("v1")
    finally: registry.MODELS, registry.MANIFEST = old
