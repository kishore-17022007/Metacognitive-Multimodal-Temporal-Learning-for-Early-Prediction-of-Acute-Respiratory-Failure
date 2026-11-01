"""Training orchestration (CPU, scikit-learn stack): baselines, ensemble, calibration, conformal, OOD, survival, explanations."""
from __future__ import annotations
import json, time, pathlib
import numpy as np, joblib
from .data.dataset import prepare, known
from .data.schema import FEATURE_NAMES
from .preprocessing.features import tabular_features, tabular_feature_names
from .models.baseline import make_baselines
from .models.survival import DiscreteTimeSurvival
from .reliability.ensemble import Ensemble
from .reliability.calibration import Calibrator
from .reliability.conformal import ConformalClassifier
from .reliability.ood import OODDetector
from .evaluation.classification import classification_metrics, pick_threshold, curves, latency_ms
from .evaluation.survival import concordance_index, brier_at
from .evaluation import reporting
from .explainability.baseline_explainer import global_importance
from .explainability.temporal_explainer import step_attribution, group_attribution, temporal_feature_attribution, temporal_differences
from .tracking.tracker import Tracker, NullTracker
from .tracking import registry
from .utils.config import path, ROOT
from .utils.reproducibility import set_seed, dict_hash, code_version
from .utils.logging import get_logger

log = get_logger()


def _tab(part): return tabular_features(part["X"], part["valid"])


def train_baselines(cfg, D=None, persist: bool = True):
    set_seed(cfg["training"]["seed"]); D = D or prepare(cfg)
    ece_bins = cfg["training"]["reliability"]["ece_bins"]; P = {k: known(v) for k, v in D["parts"].items()}
    Xtr, ytr = _tab(P["train"]), P["train"]["meta"].y.values; Xva, yva = _tab(P["validation"]), P["validation"]["meta"].y.values
    Xte, yte = _tab(P["test"]), P["test"]["meta"].y.values
    tr = Tracker("tbi-arf-baselines") if persist else NullTracker(); results = {}
    for name, m in make_baselines(cfg["model"], cfg["training"]["seed"]).items():
        with tr.start_run(name) as run:
            t0 = time.time(); m.fit(Xtr, ytr); dur = time.time() - t0
            th = pick_threshold(yva, m.predict_proba(Xva)[:, 1])                 # threshold from VALIDATION only
            val = classification_metrics(yva, m.predict_proba(Xva)[:, 1], th, ece_bins); test = classification_metrics(yte, m.predict_proba(Xte)[:, 1], th, ece_bins)
            lat = latency_ms(lambda x: m.predict_proba(x), Xte)
            run.log_params({"model": name, "seed": cfg["training"]["seed"], "dataset_version": f"synthetic-{D['raw_hash']}", "data_provenance": "synthetic",
                            "preprocessing_hash": dict_hash(D["pre"].to_dict()), "code_version": code_version(), "n_features": Xtr.shape[1], "horizon_h": D["ds"]["H"]})
            run.log_metrics({**{f"val_{k}": v for k, v in val.items() if isinstance(v, float)}, **{f"test_{k}": v for k, v in test.items() if isinstance(v, float)}, "train_seconds": dur, **{f"lat_{k}": v for k, v in lat.items()}})
        results[name] = {"validation": val, "test": test, "latency": lat, "train_seconds": dur, "curves": curves(yte, m.predict_proba(Xte)[:, 1])}
        log.info(f"baseline {name}: val AUROC={val['auroc']:.3f} test AUROC={test['auroc']:.3f}")
    if persist:
        reporting.write_json({k: {kk: vv for kk, vv in v.items() if kk != "curves"} for k, v in results.items()}, "baseline_metrics.json")
        reporting.write_json({k: v["curves"] for k, v in results.items()}, "baseline_curves.json")
    return results


def train_reliability_stack(cfg, D=None, version: str | None = None, models_root=None, persist: bool = True):
    """Ensemble + calibrator + conformal + OOD + discrete-time survival + explanations. Saves a versioned artifact dir."""
    seed = cfg["training"]["seed"]; set_seed(seed); D = D or prepare(cfg); rc = cfg["training"]["reliability"]; mc = cfg["model"]["ensemble"]
    P = {k: known(v) for k, v in D["parts"].items()}; K = D["ds"]["K"]; H = D["ds"]["H"]
    Xtr, mtr = _tab(P["train"]), P["train"]["meta"]; Xva, mva = _tab(P["validation"]), P["validation"]["meta"]
    Xca, mca = _tab(P["calibration"]), P["calibration"]["meta"]; Xte, mte = _tab(P["test"]), P["test"]["meta"]
    version = version or f"v{time.strftime('%Y%m%d-%H%M%S')}"; vdir = (pathlib.Path(models_root) if models_root else path("artifacts", "models", "x").parent) / version; vdir.mkdir(parents=True, exist_ok=True); t0 = time.time()
    tr = Tracker("tbi-arf-reliability") if persist else NullTracker()
    with tr.start_run(f"reliability-stack-{version}") as run:
        ens = Ensemble(seed=seed, n_members=mc["n_members"], max_iter=mc["max_iter"], learning_rate=mc["learning_rate"], max_depth=mc["max_depth"]).fit(Xtr, mtr.y.values, mtr.patient_id.values)
        raw_va = ens.member_probs(Xva).mean(1); raw_thr = pick_threshold(mva.y.values, raw_va)          # validation
        # split calibration partition BY PATIENT: half -> calibrator, half -> conformal
        pids = np.array(sorted(mca.patient_id.unique())); rng = np.random.default_rng(seed); rng.shuffle(pids); half = set(pids[: len(pids) // 2])
        a = mca.patient_id.isin(half).values
        raw_ca = ens.member_probs(Xca).mean(1)
        cal = Calibrator(rc["calibration_method"]).fit(raw_ca[a], mca.y.values[a])
        conf = ConformalClassifier(rc["conformal_alpha"]).fit(cal.predict(raw_ca[~a]), mca.y.values[~a])
        cal_thr = float(cal.predict(np.array([raw_thr]))[0])
        ood = OODDetector(rc["ood_flag_rate_on_validation"], seed).fit(Xtr, Xva)
        # survival (all windows incl. censored)
        PS = {k: v for k, v in D["parts"].items()}; Xs_tr = _tab(PS["train"]); mm = PS["train"]["meta"]
        surv = DiscreteTimeSurvival(K, seed, mc["max_iter"]).fit(Xs_tr, mm.surv_time.values, mm.surv_event.values)
        # ---- TEST evaluation (final) ----
        A = ens.aggregate(Xte, raw_thr); cal_te = cal.predict(A["mean"]); yte = mte.y.values
        m_raw = classification_metrics(yte, A["mean"], raw_thr, rc["ece_bins"]); m_cal = classification_metrics(yte, cal_te, cal_thr, rc["ece_bins"])
        cov, avg_size = conf.coverage(cal_te, yte); sets = conf.predict_sets(cal_te)
        s_te, f_te = ood.flag(Xte); s_va, f_va = ood.flag(Xva)
        Xs_te = _tab(PS["test"]); mt = PS["test"]["meta"]; S = surv.survival(Xs_te)
        ci = concordance_index(mt.surv_time.values, mt.surv_event.values, 1 - S[:, -1])
        briers = {f"k{k}": brier_at(mt.surv_time.values, mt.surv_event.values, S[:, k - 1], k) for k in (1, 3, K)}
        mono = bool(np.all(np.diff(S, axis=1) <= 1e-9))
        rel = {"conformal": {"alpha": conf.alpha, "qhat": conf.qhat, "empirical_coverage_test": cov, "avg_set_size": avg_size, "set_distribution": {str(s): int(sum(1 for x in sets if x == s)) for s in ([], [0], [1], [0, 1])} if False else {"empty": sum(len(x) == 0 for x in sets), "{0}": sum(x == [0] for x in sets), "{1}": sum(x == [1] for x in sets), "{0,1}": sum(len(x) == 2 for x in sets)},
                             "caveat": "marginal coverage assumes exchangeability; windows within a patient are dependent and drift breaks it"},
               "ood": {"threshold": ood.threshold, "val_flag_rate": float(f_va.mean()), "test_flag_rate": float(f_te.mean())},
               "ensemble": {"mean_std_test": float(A["std"].mean()), "mean_agreement_test": float(A["agreement"].mean()), "n_members": ens.n},
               "calibration": {"method": cal.status, "ece_raw": m_raw["ece"], "ece_calibrated": m_cal["ece"], "brier_raw": m_raw["brier"], "brier_calibrated": m_cal["brier"]}}
        surv_rep = {"c_index_test": ci, "brier": briers, "survival_monotone": mono, "n_test_windows": int(len(mt)), "n_events": int(mt.surv_event.sum()),
                    "limitations": ["Harrell C on windows (patient-dependent samples)", "Brier without IPCW", "trajectory head only available in the PyTorch multi-task model"]}
        # ---- explanations (validation for global; one test example for temporal) ----
        names, groups = tabular_feature_names()
        hgb = ens.members[0]; gi = global_importance(hgb, Xva, mva.y.values, names, groups, seed)
        fn = lambda x: ens.member_probs(x).mean(1)
        pos = np.where(yte == 1)[0]; ex = int(pos[0]) if len(pos) else 0
        Xw, Vw = P["test"]["X"][ex], P["test"]["valid"][ex]
        base, att = step_attribution(fn, Xw, Vw); _, gatt = group_attribution(fn, Xw, Vw)
        _, temporal_att = temporal_feature_attribution(fn, Xw, Vw); temporal_diff = temporal_differences(Xw, Vw)
        if persist: reporting.plot_temporal(att, Vw)
        expl = {"global": gi, "example_window": {"patient_id_hash": dict_hash({"p": str(mte.patient_id.iloc[ex])}), "baseline_risk": base, "step_attribution": att.tolist(), "group_attribution": gatt,
            "temporal_feature_attribution": temporal_att.tolist(), "temporal_feature_differences": temporal_diff.tolist(), "feature_names": FEATURE_NAMES,
            "relative_steps": (np.arange(len(Vw)) - len(Vw) + 1).tolist(), "valid_steps": Vw.tolist(),
                "label": int(yte[ex])}, "limitations": ["Attribution is not causal", "Attention/occlusion can be unstable across windows", "Synthetic explanations reflect the synthetic generator"]}
        # ---- save artifacts ----
        joblib.dump(ens, vdir / "ensemble.joblib"); joblib.dump(cal, vdir / "calibrator.joblib"); joblib.dump(ood, vdir / "ood.joblib"); joblib.dump(surv, vdir / "survival.joblib")
        D["pre"].save(vdir / "preprocessor.json")
        thr = {"raw_threshold": raw_thr, "calibrated_threshold": cal_thr, "horizon_hours": H, "n_intervals": K, "window_length": D["ds"]["X"].shape[1]}
        (vdir / "thresholds.json").write_text(json.dumps(thr)); (vdir / "conformal.json").write_text(json.dumps(conf.to_dict()))
        meta = {"version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "data_provenance": "synthetic", "dataset_version": f"synthetic-{D['raw_hash']}", "seed": seed, "code_version": code_version(),
                "profile": cfg["profile"], "disclaimer": "Research prototype on SYNTHETIC data. Not a validated clinical tool.", "reliability_config": rc,
                "test_metrics_calibrated": m_cal, "test_metrics_raw": m_raw, "train_seconds": time.time() - t0}
        (vdir / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
        files = ["ensemble.joblib", "calibrator.joblib", "ood.joblib", "survival.joblib", "preprocessor.json", "thresholds.json", "conformal.json", "metadata.json"]
        if persist: registry.register(version, files, {"auroc_test": m_cal["auroc"], "auprc_test": m_cal["auprc"], "data_provenance": "synthetic"})
        # curves/plots for dashboard
        cv = {"ensemble_calibrated": curves(yte, cal_te)}
        if persist: reporting.plot_calibration(yte, A["mean"], cal_te, rc["ece_bins"])
        out = {"version": version, "metrics": {"raw": m_raw, "calibrated": m_cal}, "reliability": rel, "survival": surv_rep, "explainability": expl, "curves": cv,
               "latency": latency_ms(lambda x: ens.member_probs(x), Xte), "thresholds": thr}
        if persist: reporting.write_json(out, "advanced_report.json")
        run.log_params({"version": version, "seed": seed, "n_members": ens.n, "calibration": cal.status, "conformal_alpha": conf.alpha, "dataset_version": meta["dataset_version"], "data_provenance": "synthetic", "code_version": meta["code_version"]})
        run.log_metrics({f"test_{k}": v for k, v in m_cal.items() if isinstance(v, float)} | {"conformal_coverage": cov, "ood_test_flag_rate": float(f_te.mean()), "survival_c_index": ci if ci else float("nan"), "train_seconds": time.time() - t0})
        run.log_artifact(vdir / "metadata.json")
    log.info(f"reliability stack {version}: test AUROC={m_cal['auroc']:.3f} AUPRC={m_cal['auprc']:.3f} conformal coverage={cov:.3f}")
    return out
