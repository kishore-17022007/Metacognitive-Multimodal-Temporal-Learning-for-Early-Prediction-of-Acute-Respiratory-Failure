import numpy as np
import pandas as pd

from tbi_mlops.evaluation.research import (
    _event_warning_metrics,
    _metrics_for_trajectory,
    _stress_inputs,
    binary_metrics,
)


def test_binary_metrics_empty_and_single_class_are_explicit():
    empty = binary_metrics([], [], 0.5)
    assert empty["status"] == "no_valid_targets"
    assert empty["n"] == 0 and empty["auroc"] is None

    one_class = binary_metrics([0, 0], [0.1, 0.8], 0.5)
    assert one_class["status"] == "single_class_labels"
    assert one_class["auroc"] is None and one_class["auprc"] is None
    assert np.isfinite(one_class["brier"])


def test_trajectory_metrics_respect_valid_target_mask():
    meta = pd.DataFrame({
        "traj_0": [0, 1, 1],
        "trajvalid_0": [1, 1, 0],
        "traj_1": [0, 1, 0],
        "trajvalid_1": [1, 0, 0],
    })
    probability = np.array([[0.1, 0.1], [0.9, 0.9], [0.99, 0.99]])
    result = _metrics_for_trajectory(meta, probability, bins=10, interval_hours=1.0)
    assert result[0]["metrics"]["n"] == 2
    assert result[1]["metrics"]["n"] == 1
    assert result[1]["metrics"]["status"] == "single_class_labels"


def test_stress_inputs_are_reproducible_and_do_not_mutate_test_array():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(4, 3, 47)).astype(np.float32)
    valid = np.array([[False, True, True]] * 4)
    original = X.copy()
    first = _stress_inputs(X, valid, seed=123)
    second = _stress_inputs(X, valid, seed=123)
    assert np.array_equal(X, original)
    assert first.keys() == second.keys()
    for name in first:
        assert np.array_equal(first[name]["X"], second[name]["X"])
        assert not np.shares_memory(first[name]["X"], X)
    for name in ("random_vital_missingness_30pct", "vital_measurement_noise"):
        changed = first[name]["X"] != X
        assert not changed[:, 0].any()
    missing = first["random_vital_missingness_30pct"]["X"]
    assert np.all(missing[:, :, 15:30][missing[:, :, 15:30] == 0] == 0)


def test_event_warning_metrics_respect_known_censoring_and_count_alerts():
    meta = pd.DataFrame({
        "patient_id": ["event", "event", "negative", "negative", "censored"],
        "t_index_h": [1.0, 3.0, 1.0, 2.0, 1.0],
        "label_known": [True, True, True, False, False],
    })
    patients = pd.DataFrame({
        "patient_id": ["event", "negative", "censored"],
        "outcome_known": [True, True, True],
        "event_observed": [True, False, False],
        "event_time_h": [4.0, np.nan, np.nan],
    })
    result = _event_warning_metrics(
        meta, np.array([0.8, 0.9, 0.9, 0.9, 0.9]), patients,
        {"event", "negative", "censored"}, threshold=0.5,
    )
    assert result["eligible_events"] == 1
    assert result["detected_events"] == 1
    assert result["missed_events"] == 0
    assert result["false_alert_patients"] == 1
    assert result["false_alert_windows"] == 1
    assert result["lead_time_distribution_hours"]["median_hours"] == 3.0