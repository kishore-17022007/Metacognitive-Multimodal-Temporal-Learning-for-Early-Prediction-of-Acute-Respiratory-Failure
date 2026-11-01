import pandas as pd, streamlit as st
from tbi_mlops.tracking.tracker import list_runs, HAS_MLFLOW


def render():
    st.title("Experiment tracking"); runs = list_runs(); st.caption(f"Backend: {'MLflow + local JSON' if HAS_MLFLOW else 'local JSON store (MLflow not installed)'} · `mlflow ui --backend-store-uri ./mlruns` for the full UI")
    if not runs: st.info("No runs yet. Run the training scripts."); return
    df = pd.DataFrame([{"experiment": r["experiment"], "run": r["name"], "status": r["status"], "duration_s": round(r["duration_s"], 1), **{k: round(v, 4) for k, v in r["metrics"].items() if k.startswith("test_")}} for r in runs]); st.dataframe(df)
    sel = st.selectbox("Run details", [f"{r['experiment']} / {r['name']} / {r['run_id']}" for r in runs]); r = next(x for x in runs if x["run_id"] == sel.split(" / ")[-1])
    st.subheader("Parameters"); st.json(r["params"]); st.subheader("Metrics"); st.json(r["metrics"]); st.subheader("Artifacts"); st.write(r["artifacts"])
