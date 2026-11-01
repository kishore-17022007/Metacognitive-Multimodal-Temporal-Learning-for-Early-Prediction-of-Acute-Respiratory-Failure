"""End-to-end local verification. Runs the whole pipeline and asserts on real outputs. Exit code 0 only if all checks pass.
Optional components (torch/fastapi/streamlit/mlflow) are reported as SKIPPED when not installed - never as passed."""
import sys, subprocess, pathlib, json, importlib.util, time
ROOT = pathlib.Path(__file__).resolve().parents[1]; PY = sys.executable; results = []


def run(label, *cmd):
    t = time.time(); r = subprocess.run([PY, *cmd], cwd=ROOT, capture_output=True, text=True)
    ok = r.returncode == 0; results.append((label, "PASS" if ok else "FAIL", f"{time.time()-t:.0f}s"))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if not ok: print(r.stdout[-1500:], r.stderr[-1500:])
    return ok


def check(label, cond, detail=""):
    results.append((label, "PASS" if cond else "FAIL", detail)); print(f"[{'PASS' if cond else 'FAIL'}] {label} {detail}"); return cond


rep = ROOT / "artifacts" / "reports"
steps = [("generate synthetic data", "-m", "src.tbi_mlops.data.generate_synthetic"), ("validate + preprocess", "scripts/prepare_data.py"), ("train baselines", "scripts/train_baseline.py"),
         ("train reliability stack", "scripts/train_advanced.py"), ("streaming/drift simulation", "simulation/run_simulation.py"), ("evaluation report", "scripts/evaluate.py"), ("automated tests", "scripts/run_tests.py")]
good = all([run(*s) for s in steps])
if good:
    adv = json.load(open(rep / "advanced_report.json")); m = adv["metrics"]["calibrated"]; r = adv["reliability"]
    check("test AUROC above chance on synthetic labels", m["auroc"] > 0.7, f"auroc={m['auroc']:.3f}")
    check("conformal coverage near target", r["conformal"]["empirical_coverage_test"] >= 1 - r["conformal"]["alpha"] - 0.05, f"cov={r['conformal']['empirical_coverage_test']:.3f}")
    check("survival monotone + C-index computed", adv["survival"]["survival_monotone"] and adv["survival"]["c_index_test"] is not None)
    sim = json.load(open(rep / "drift_simulation.json")); check("normal scenario no drift; drift scenarios alert", sim["normal"]["drift_overall"] == "ok" and all(sim[s]["drift_overall"] == "alert" for s in ("abrupt_drift", "gradual_drift", "abnormal_inputs")))
    check("experiment report written", (rep / "EXPERIMENT_REPORT.md").exists()); check("local run tracking recorded", any((ROOT / "mlruns_local").rglob("*.json")))
    man = json.load(open(ROOT / "artifacts" / "models" / "manifest.json")); check("model registered as candidate, none silently promoted", len(man["versions"]) >= 1 and man["production"] is None)
for lib, what in (("torch", "PyTorch Transformer/Graph/multi-task"), ("fastapi", "FastAPI service"), ("streamlit", "Streamlit dashboard"), ("mlflow", "MLflow backend")):
    if importlib.util.find_spec(lib) is None: results.append((f"{what} execution", "SKIPPED", f"{lib} not installed")); print(f"[SKIP] {what}: {lib} not installed -> code present but NOT verified here")
print("\n=== SUMMARY ==="); [print(f"{s:8s} {l} {d}") for l, s, d in results]
sys.exit(0 if all(s != "FAIL" for _, s, _ in results) else 1)
