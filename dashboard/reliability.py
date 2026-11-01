import pandas as pd, streamlit as st
from .common import *


def render():
    st.title("Reliability center"); adv = need(report("advanced_report.json")); r = adv["reliability"]
    c = st.columns(3); c[0].metric("Ensemble mean std (test)", f"{r['ensemble']['mean_std_test']:.3f}"); c[1].metric("Member agreement", f"{r['ensemble']['mean_agreement_test']:.2f}"); c[2].metric("OOD test flag rate", f"{r['ood']['test_flag_rate']:.3f}")
    st.subheader("Calibration"); st.write(r["calibration"])
    if (GRAPHS / "calibration.png").exists(): st.image(str(GRAPHS / "calibration.png"))
    st.subheader("Conformal prediction"); st.write(r["conformal"]); st.warning(r["conformal"]["caveat"])
    sim = report("drift_simulation.json")
    if sim: st.subheader("Reliability under synthetic scenarios"); st.dataframe(pd.DataFrame(sim).T[["ood_rate", "mean_ensemble_std", "mean_risk", "drift_overall"]])
    e = st.session_state.get("engine")
    st.subheader("Alert history (current simulation)"); st.dataframe(pd.DataFrame(e.alert_history) if e and e.alert_history else pd.DataFrame({"info": ["no alerts in this session"]}))
