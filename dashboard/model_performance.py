import pandas as pd, plotly.graph_objects as go, streamlit as st
from .common import *


def render():
    st.title("Model evaluation (test partition, synthetic)")
    adv, base, dq = need(report("advanced_report.json")), report("baseline_metrics.json"), report("data_quality_report.json")
    a = adv["metrics"]["calibrated"]; k = st.columns(6); [k[i].metric(n.upper(), f"{a[n]:.3f}") for i, n in enumerate(["auroc", "auprc", "precision", "recall", "f1", "brier"])]
    cm = a["confusion_matrix"]; st.write(f"Confusion matrix @ validation-selected threshold {a['threshold']:.3f}"); st.dataframe(pd.DataFrame(cm, index=["true 0", "true 1"], columns=["pred 0", "pred 1"]))
    c = adv["curves"]["ensemble_calibrated"]; l, r = st.columns(2)
    f = go.Figure(go.Scatter(x=c["fpr"], y=c["tpr"])); f.update_layout(title="ROC", template="plotly_dark", height=320); l.plotly_chart(f, use_container_width=True)
    f = go.Figure(go.Scatter(x=c["recall"], y=c["precision"])); f.update_layout(title="Precision-Recall", template="plotly_dark", height=320); r.plotly_chart(f, use_container_width=True)
    if (GRAPHS / "calibration.png").exists(): st.image(str(GRAPHS / "calibration.png"))
    st.subheader("Survival evaluation"); s = adv["survival"]; st.write({"c_index": s["c_index_test"], "brier": s["brier"], "events": s["n_events"]}); st.caption("; ".join(s["limitations"]))
    if base:
        st.subheader("Model comparison"); comparison = {n: {m: v["test"].get(m) for m in ("auroc", "auprc", "f1", "brier", "ece")} for n, v in base.items()}
        comparison["bootstrap ensemble (calibrated)"] = {m: a.get(m) for m in ("auroc", "auprc", "f1", "brier", "ece")}
        torch = report("torch_report.json")
        if torch:
            comparison["PyTorch Transformer + physiological GAT"] = {"auroc": torch["test"].get("auroc"), "auprc": torch["test"].get("auprc"), "f1": None, "brier": None, "ece": None}
        st.dataframe(pd.DataFrame(comparison).T)
        st.subheader("ROC and precision-recall curves: all trained models")
        model_curves = report("model_curves.json") or {}
        if model_curves:
            left, right = st.columns(2); roc = go.Figure(); pr = go.Figure()
            for name, curve in model_curves.items():
                roc.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], name=name, mode="lines"))
                pr.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], name=name, mode="lines"))
            roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="chance", mode="lines", line=dict(dash="dash", color="gray")))
            roc.update_layout(title="ROC (held-out synthetic test partition)", xaxis_title="False positive rate", yaxis_title="True positive rate", template="plotly_dark", height=420)
            pr.update_layout(title="Precision-recall (held-out synthetic test partition)", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark", height=420)
            left.plotly_chart(roc, use_container_width=True); right.plotly_chart(pr, use_container_width=True)
        else:
            st.info("Per-model curves are not available yet. Run scripts/train_baseline.py, scripts/train_advanced.py, then scripts/evaluate.py.")
    if dq: st.subheader("Split information"); st.dataframe(pd.DataFrame(dq["partition_stats"]).T)
