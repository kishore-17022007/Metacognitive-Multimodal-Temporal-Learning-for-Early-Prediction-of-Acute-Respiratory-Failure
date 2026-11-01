"""Streamlit dashboard. [UNTESTED in build sandbox: streamlit/plotly unavailable; py_compile only]
Run: streamlit run app.py   (Reads real artifacts produced by the training scripts; shows empty states otherwise.)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
import streamlit as st
from dashboard import overview, patient_simulation, model_performance, reliability, experiments, explainability, monitoring_page

st.set_page_config(page_title="TBI ARF Research Platform", layout="wide", page_icon="🫁")
st.markdown("""<style>
.block-container{padding-top:1.5rem} .banner{background:#2b2233;border-left:4px solid #d9a441;padding:.5rem .8rem;border-radius:4px;font-size:.85rem}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="banner">Research prototype on <b>synthetic data</b>. Risk values are estimates, not diagnoses. Not a validated clinical tool.</div>', unsafe_allow_html=True)
PAGES = {"1 · System overview": overview, "2 · Patient streaming simulation": patient_simulation, "3 · Model evaluation": model_performance,
         "4 · Explainability": explainability, "5 · Reliability center": reliability, "6 · MLOps monitoring": monitoring_page, "7 · Experiment tracking": experiments}
choice = st.sidebar.radio("Page", list(PAGES)); PAGES[choice].render()
