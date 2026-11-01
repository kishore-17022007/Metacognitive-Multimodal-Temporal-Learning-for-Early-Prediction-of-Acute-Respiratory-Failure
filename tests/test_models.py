import unittest
from helpers import *
from tbi_mlops.models.survival import DiscreteTimeSurvival, expand
from tbi_mlops.models.baseline import make_baselines
from tbi_mlops.evaluation.survival import concordance_index, brier_at
from tbi_mlops.preprocessing.features import tabular_features, tabular_feature_names
from tbi_mlops.data.dataset import known


def test_tabular_dimensions():
    ds = tiny()["D"]["ds"]; T = tabular_features(ds["X"], ds["valid"]); assert T.shape == (len(ds["X"]), len(tabular_feature_names()[0])) and np.isfinite(T).all()


def test_baselines_train_and_predict():
    t = tiny(); P = known(t["D"]["parts"]["train"]); X = tabular_features(P["X"], P["valid"]); y = P["meta"].y.values
    for n, m in make_baselines(t["cfg"]["model"]).items():
        m.fit(X, y); p = m.predict_proba(X[:20])[:, 1]; assert p.shape == (20,) and ((p >= 0) & (p <= 1)).all(), n


def test_survival_censoring_expansion():
    X = np.zeros((3, 2), np.float32); Xp, yp = expand(X, np.array([3, 2, 4]), np.array([1, 0, 0]), 6)
    assert len(yp) == 3 + 2 + 4 and yp.tolist() == [0, 0, 1, 0, 0, 0, 0, 0, 0], "event only at its interval; censored rows never positive"


def test_survival_probabilities_consistent():
    t = tiny(); P = t["D"]["parts"]["test"]; S = t["P"].surv.survival(tabular_features(P["X"], P["valid"]))
    assert S.shape[1] == t["D"]["ds"]["K"] and ((S >= 0) & (S <= 1)).all() and (np.diff(S, axis=1) <= 1e-9).all()


def test_cindex_and_brier_toy():
    time = np.array([1, 2, 3, 4]); ev = np.array([1, 1, 1, 0]); assert concordance_index(time, ev, np.array([4, 3, 2, 1])) == 1.0
    assert concordance_index(time, ev, np.array([1, 2, 3, 4])) == 0.0; assert concordance_index(np.array([1, 2]), np.array([0, 0]), np.array([1, 2])) is None
    b = brier_at(np.array([1] * 5 + [6] * 20), np.array([1] * 5 + [0] * 20), np.array([0.0] * 5 + [1.0] * 20), 3); assert b["brier"] == 0.0


def test_survival_evaluated_on_test_with_events():
    s = tiny()["out"]["survival"]; assert s["n_events"] > 0 and s["c_index_test"] is not None and s["survival_monotone"]


def test_temporal_feature_explanations_match_window_and_mask_padding():
    from tbi_mlops.explainability.temporal_explainer import temporal_feature_attribution, temporal_differences
    from tbi_mlops.data.schema import N_F
    t = tiny(); part = t["D"]["parts"]["test"]; x, valid = part["X"][0], part["valid"][0]
    _, attribution = temporal_feature_attribution(t["P"].model_fn, x, valid); differences = temporal_differences(x, valid)
    assert attribution.shape == differences.shape == (len(valid), N_F)
    assert np.isfinite(attribution).all() and np.all(attribution[~valid] == 0)
    assert np.isnan(differences[~valid]).all()


def test_baseline_beats_chance_on_synthetic_labels():
    out = tiny()["out"]; assert out["metrics"]["calibrated"]["auroc"] > 0.7


def _torch():
    try: import torch; return torch
    except ImportError: raise unittest.SkipTest("torch not installed (torch models UNTESTED in build sandbox)")


def test_torch_forward_cpu_and_shapes():
    torch = _torch(); from tbi_mlops.models.multitask import MultiTaskModel
    ds = tiny()["D"]["ds"]; m = MultiTaskModel(ds["X"].shape[2], ds["K"], ds["X"].shape[1], tiny()["cfg"]["model"]); o = m(torch.tensor(ds["X"][:8]), torch.tensor(ds["valid"][:8]))
    assert o["binary"].shape == (8,) and o["traj"].shape == (8, ds["K"]) and o["haz"].shape == (8, ds["K"])


def test_torch_attention_masks_and_padding_are_respected():
    torch = _torch(); from tbi_mlops.models.physiological_graph import GATLayer, adjacency
    from tbi_mlops.models.temporal_transformer import TemporalTransformer
    A = adjacency(); layer = GATLayer(8, heads=2, dropout=0); _, attention = layer(torch.randn(3, len(A), 8), A)
    assert attention.shape == (3, len(A), len(A), 2)
    assert torch.allclose(attention.sum(dim=2), torch.ones(3, len(A), 2), atol=1e-6)
    assert torch.count_nonzero(attention.masked_select((A == 0)[None, :, :, None])) == 0
    ds = tiny()["D"]["ds"]; valid = torch.tensor(ds["valid"][:4]); x = torch.tensor(ds["X"][:4]); model = TemporalTransformer(x.shape[2], x.shape[1]).eval()
    padded_variant = x.clone(); padded_variant[~valid] = 1000
    assert torch.allclose(model(x, valid), model(padded_variant, valid), atol=1e-6)


def test_torch_training_step_checkpoint_and_survival_loss():
    torch = _torch(); from tbi_mlops.models.multitask import MultiTaskModel, multitask_loss, survival_nll, survival_from_logits
    import tempfile, os
    t = tiny(); D = t["D"]; ds = D["ds"]; K = ds["K"]; m = MultiTaskModel(ds["X"].shape[2], K, ds["X"].shape[1], t["cfg"]["model"]); me = ds["meta"].iloc[:16]
    b = {"y": torch.tensor(me.y.values), "label_known": torch.tensor(me.label_known.values), "traj": torch.tensor(me[[f"traj_{k}" for k in range(K)]].values, dtype=torch.float32),
         "traj_valid": torch.tensor(me[[f"trajvalid_{k}" for k in range(K)]].values, dtype=torch.float32), "surv_time": torch.tensor(me.surv_time.values), "surv_event": torch.tensor(me.surv_event.values)}
    loss, parts = multitask_loss(m(torch.tensor(ds["X"][:16]), torch.tensor(ds["valid"][:16])), b, t["cfg"]["model"]["multitask"]); loss.backward(); assert torch.isfinite(loss)
    p = os.path.join(tempfile.mkdtemp(), "m.pt"); torch.save(m.state_dict(), p); m2 = MultiTaskModel(ds["X"].shape[2], K, ds["X"].shape[1], t["cfg"]["model"]); m2.load_state_dict(torch.load(p, weights_only=True))
    m.eval(); m2.eval(); x, v = torch.tensor(ds["X"][:4]), torch.tensor(ds["valid"][:4]); assert torch.allclose(m(x, v)["binary"], m2(x, v)["binary"], atol=1e-6)
    lg = torch.tensor([[0.0, 1.0, 0.0]])
    event_nll = survival_nll(lg, torch.tensor([2]), torch.tensor([1]))
    censored_nll = survival_nll(lg, torch.tensor([2]), torch.tensor([0]))
    assert torch.allclose(event_nll, -torch.nn.functional.logsigmoid(lg[0, 0]) - torch.nn.functional.logsigmoid(lg[0, 1]))
    assert torch.allclose(censored_nll, -torch.nn.functional.logsigmoid(-lg[0, 0]) - torch.nn.functional.logsigmoid(-lg[0, 1]))
    assert event_nll != censored_nll
    assert (survival_from_logits(torch.randn(5, 6)).diff(dim=1) <= 1e-6).all()
