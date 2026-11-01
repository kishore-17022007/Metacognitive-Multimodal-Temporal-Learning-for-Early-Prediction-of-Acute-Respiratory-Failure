import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from .common import *
from tbi_mlops.streaming.engine import StreamEngine
from tbi_mlops.streaming.replay import SCENARIOS
from tbi_mlops.data.schema import FEATURES, GROUPS

try: from streamlit_autorefresh import st_autorefresh
except ImportError: st_autorefresh = None


def render():
    st.title("Patient streaming simulation (synthetic replay)")
    P, clean, pat = get_predictor(), clean_obs(), patients()
    if isinstance(P, str) or clean is None: need(None)
    cfg = load_config(); ids = sorted(clean.patient_id.unique())
    c = st.columns([2, 2, 2]); pid = c[0].selectbox("Simulated patient", ids[:200]); sc = c[1].selectbox("Scenario", SCENARIOS); speed = c[2].slider("Replay speed (sim s / wall s)", 60, 7200, 1800)
    key = (pid, sc)
    if st.session_state.get("eng_key") != key:
        row = pat[pat.patient_id == pid].iloc[0]; ev = row.event_time_h if bool(row.event_observed) else None
        st.session_state.engine = StreamEngine(P, clean[clean.patient_id == pid], sc, ev, reference=clean, monitoring_cfg=cfg["monitoring"]); st.session_state.eng_key = key; st.session_state.last = time.time()
    e = st.session_state.engine; b = st.columns(5)
    if b[0].button("Start"): e.start(); st.session_state.last = time.time()
    if b[1].button("Pause"): e.pause()
    if b[2].button("Resume"): e.resume(); st.session_state.last = time.time()
    if b[3].button("Step"): e.step()
    if b[4].button("Reset"): e.reset()
    now = time.time(); e.tick(now - st.session_state.get("last", now), speed); st.session_state.last = now
    st.metric("Simulated clock (h since admission)", f"{e.clock_h:.1f}")
    if not e.predictions: st.info("Press Start or Step."); return
    import pandas as pd
    rows = pd.DataFrame(e.rows); t = rows.hours_since_admission
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=[g.title() for g in GROUPS], vertical_spacing=0.05)
    for gi, g in enumerate(GROUPS, 1):
        for f in FEATURES:
            if f.group == g and f.name not in ("sedated", "pupil_response"): fig.add_trace(go.Scatter(x=t, y=rows[f.name], name=f.name, mode="lines+markers", marker_size=4), row=gi, col=1)
    fig.update_layout(height=700, template="plotly_dark", margin=dict(t=30, b=10)); st.plotly_chart(fig, use_container_width=True)
    pr = pd.DataFrame([{k: p[k] for k in ("sim_hour", "predicted_probability", "prediction_status", "inference_latency_ms")} | {"std": p["uncertainty_summary"]["ensemble_std"], "ood": p["ood_status"]["flag"], "set": str(p["conformal_set"])} for p in e.predictions])
    f2 = go.Figure(); f2.add_trace(go.Scatter(x=pr.sim_hour, y=pr.predicted_probability, name="estimated risk", line_color="#e8a33d"))
    f2.add_trace(go.Scatter(x=pr.sim_hour, y=(pr.predicted_probability + pr["std"]).clip(0, 1), line=dict(width=0), showlegend=False)); f2.add_trace(go.Scatter(x=pr.sim_hour, y=(pr.predicted_probability - pr["std"]).clip(0, 1), fill="tonexty", fillcolor="rgba(232,163,61,.2)", line=dict(width=0), name="± ensemble std"))
    if e.event_logged: f2.add_vline(x=e.event_time_h, line_dash="dash", annotation_text="simulated event")
    f2.update_layout(height=300, template="plotly_dark", yaxis_title=f"P(ARF within {e.predictions[-1]['prediction_horizon_hours']} h)  — estimate, not a diagnosis"); st.plotly_chart(f2, use_container_width=True)
    last = e.predictions[-1]; k = st.columns(4); k[0].metric("Status", last["prediction_status"]); k[1].metric("Conformal set", str(last["conformal_set"])); k[2].metric("OOD", "flag" if last["ood_status"]["flag"] else "no"); k[3].metric("Latency ms", f"{last['inference_latency_ms']:.0f}")
    st.write("Data-quality flags:", last["data_quality_flags"] or "none"); st.write("Reasons:", last["reliability_reasons"] or "none")
    with st.expander("Event timeline / log"): st.dataframe(pd.DataFrame([x.to_dict() for x in e.events[-100:]][::-1]))
    if e.running and st_autorefresh: st_autorefresh(interval=1000, key="sim_refresh")
    elif e.running: st.caption("Install `streamlit-autorefresh` for automatic updates; otherwise press Step / rerun.")
