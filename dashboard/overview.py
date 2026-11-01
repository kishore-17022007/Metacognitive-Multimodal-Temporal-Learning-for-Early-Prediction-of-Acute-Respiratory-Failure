import streamlit as st
from .common import *
from tbi_mlops.tracking import registry


def render():
    st.title("Metacognitive multimodal ARF early-warning — research & MLOps platform")
    adv, dq = report("advanced_report.json"), report("data_quality_report.json")
    c = st.columns(4); m = registry.manifest()
    c[0].metric("Model version", registry.latest() or "none"); c[1].metric("Data provenance", "synthetic" if dq else "n/a"); c[2].metric("Production-designated", m["production"] or "none (candidate only)")
    p = get_predictor(); c[3].metric("Service health", "ok" if not isinstance(p, str) else "no model")
    if adv:
        a = adv["metrics"]["calibrated"]; st.subheader("Evaluation summary (test, synthetic)")
        k = st.columns(5); [k[i].metric(n, f"{a[n]:.3f}") for i, n in enumerate(["auroc", "auprc", "f1", "brier", "ece"])]
        st.caption(f"Model created: {adv.get('version')} · horizon {adv['thresholds']['horizon_hours']} h")
    else: st.info("No evaluation yet. Run `python scripts/train_advanced.py`.")
    ev = report("drift_simulation.json")
    st.subheader("Recent reliability / drift alerts")
    st.write({k: v["drift_overall"] for k, v in ev.items()} if ev else "No simulation run yet (python simulation/run_simulation.py).")
    ss = st.session_state.get("engine")
    st.subheader("Simulation status"); st.write("running" if ss and ss.running else "idle")
