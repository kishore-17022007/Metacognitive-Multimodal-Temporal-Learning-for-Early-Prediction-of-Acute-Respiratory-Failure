import numpy as np, pandas as pd, plotly.express as px, streamlit as st
from .common import *


def render():
    st.title("Explainability"); adv = need(report("advanced_report.json")); ex = adv["explainability"]
    g = pd.DataFrame(ex["global"]["features"][:15]); st.subheader("Global feature attribution"); st.caption(ex["global"]["method"])
    st.plotly_chart(px.bar(g[::-1], x="importance", y="name", orientation="h", template="plotly_dark", height=450), use_container_width=True)
    gi = ex["global"]["group_importance"]; st.subheader("Feature-group contribution"); st.plotly_chart(px.bar(x=list(gi), y=list(gi.values()), template="plotly_dark", height=260), use_container_width=True)
    w = ex["example_window"]; st.subheader("Temporal importance — example test window"); st.caption(f"baseline risk {w['baseline_risk']:.3f}, true label {w['label']} (pseudonymised id {w['patient_id_hash']})")
    att = np.array(w["step_attribution"]); st.plotly_chart(px.bar(x=np.arange(len(att)) - len(att) + 1, y=att, template="plotly_dark", height=280, labels={"x": "steps before prediction index", "y": "Δ risk if step removed"}), use_container_width=True)
    temporal = np.asarray(w.get("temporal_feature_attribution", []), dtype=float)
    differences = np.asarray(w.get("temporal_feature_differences", []), dtype=float)
    if temporal.size:
        st.subheader("Temporal feature sensitivity")
        st.caption("Each cell is baseline risk minus risk after masking that feature at that timestep. This is model sensitivity, not a causal effect.")
        st.plotly_chart(px.imshow(temporal.T, x=w["relative_steps"], y=w["feature_names"], color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect="auto", labels={"x": "steps before prediction index", "y": "feature", "color": "Δ risk"}, title="Feature-by-timestep occlusion attribution", height=620), use_container_width=True)
    if differences.size:
        st.subheader("Observed temporal differences")
        st.caption("Adjacent observed value changes in preprocessed feature units; blank cells indicate padding or a missing measurement.")
        st.plotly_chart(px.imshow(differences.T, x=w["relative_steps"], y=w["feature_names"], color_continuous_scale="RdBu", color_continuous_midpoint=0, aspect="auto", labels={"x": "steps before prediction index", "y": "feature", "color": "Δ scaled value"}, title="Adjacent-step feature changes", height=620), use_container_width=True)
    st.write("Group occlusion:", w["group_attribution"]); st.warning("\n".join(f"• {x}" for x in ex["limitations"]))
