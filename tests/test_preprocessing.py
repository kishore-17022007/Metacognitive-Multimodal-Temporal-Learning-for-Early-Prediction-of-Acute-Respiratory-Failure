from helpers import *
from tbi_mlops.data.schema import *
from tbi_mlops.preprocessing.pipeline import Preprocessor, channel_names, n_channels
from tbi_mlops.preprocessing.features import tabular_feature_names, tabular_features


def test_window_shapes_and_padding():
    ds = tiny()["D"]["ds"]; X, V = ds["X"], ds["valid"]; assert X.shape[1:] == (12, n_channels()) and len(channel_names()) == n_channels()
    assert V[:, -1].all(), "last step (prediction index) always valid"
    assert np.all(X[~V] == 0), "padding is zeros but masked by valid"; assert np.isfinite(X).all()


def test_patient_level_split_isolation():
    D = tiny()["D"]; s = D["splits"]; names = list(s)
    for i, a in enumerate(names):
        for b in names[i + 1:]: assert not (s[a] & s[b])
    for k, p in D["parts"].items(): assert set(p["meta"].patient_id) <= s[k]


def test_preprocessing_fit_on_train_only():
    D = tiny()["D"]; clean, tr = D["clean"], D["splits"]["train"]
    exp = clean[clean.patient_id.isin(tr)][FEATURE_NAMES].mean().fillna(0).values
    assert np.allclose(D["pre"].mean, exp); full = clean[FEATURE_NAMES].mean().fillna(0).values; assert not np.allclose(D["pre"].mean, full)


def test_no_future_information_in_windows():
    """Changing observations AFTER index i must not change the window at i."""
    t = tiny(); pre = t["D"]["pre"]; g, _ = long_patient("nonevent"); g = g.sort_values(TIME_COL).reset_index(drop=True); i = 15
    Z1 = pre.patient_steps(g); g2 = g.copy(); g2.loc[i + 1:, FEATURE_NAMES] = g2.loc[i + 1:, FEATURE_NAMES] * 3 + 5
    Z2 = pre.patient_steps(g2); assert np.allclose(pre.window_at(Z1, i)[0], pre.window_at(Z2, i)[0])
    # and incremental history gives the same window as the full-history computation
    Zp = pre.patient_steps(g.iloc[: i + 1]); assert np.allclose(pre.window_at(Zp, i)[0], pre.window_at(Z1, i)[0])


def test_no_outcome_derived_inputs_and_window_before_event():
    t = tiny(); banned = ("event", "outcome", "followup", "surv", "label", "censor")
    names, _ = tabular_feature_names(); assert not any(b in n for n in names + channel_names() for b in banned)
    m = t["D"]["ds"]["meta"].merge(t["pat"][["patient_id", "event_observed", "event_time_h"]], on="patient_id")
    ev = m[m.event_observed]; assert (ev.t_index_h < ev.event_time_h).all()


def test_missingness_indicators_and_masking():
    t = tiny(); ds = t["D"]["ds"]; F = N_F; X = ds["X"]
    assert set(np.unique(X[:, :, F:2 * F])) <= {0.0, 1.0}; assert (X[:, :, F:2 * F][ds["valid"]] == 0).any(), "missingness is represented"
    sparse = FEATURE_NAMES.index("lactate"); assert X[:, -1, F + sparse].mean() < 0.6, "labs are sparse and flagged as such"


def test_preprocessor_roundtrip(tmp_path=None):
    import tempfile, os
    pre = tiny()["D"]["pre"]; p = os.path.join(tempfile.mkdtemp(), "p.json"); pre.save(p); q = Preprocessor.load(p)
    assert np.allclose(pre.mean, q.mean) and np.allclose(pre.std, q.std) and np.allclose(pre.median, q.median)
