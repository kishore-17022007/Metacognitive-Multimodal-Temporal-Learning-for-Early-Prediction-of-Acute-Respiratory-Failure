import pandas as pd, plotly.express as px, streamlit as st
from .common import *
from tbi_mlops.tracking import registry


def render():
    st.title("MLOps monitoring"); sim = need(report("drift_simulation.json"), "python simulation/run_simulation.py")
    sc = st.selectbox("Scenario", list(sim)); st.write(sim[sc]); df = pd.DataFrame(sim).T
    st.plotly_chart(px.bar(df.reset_index(), x="index", y="n_features_alert", template="plotly_dark", title="Drifted features (PSI alert) by scenario"), use_container_width=True)
    dp = report("delayed_performance.json"); st.subheader("Delayed (label-matured) performance")
    if dp: st.write(dp["delayed_performance"]); st.caption("Only predictions whose horizon has elapsed are scored; pending ones are never evaluated.")
    e = st.session_state.get("engine")
    if e: st.subheader("Live simulation ops"); st.write(e.ops.summary())
    m = registry.manifest(); st.subheader("Model version"); st.write({"latest": registry.latest(), "production": m["production"]})
