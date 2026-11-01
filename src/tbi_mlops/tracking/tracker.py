"""Experiment tracking. Uses MLflow when installed (UNTESTED in build sandbox); otherwise a local JSON run store with the
same interface at ./mlruns_local (TESTED). Dashboard reads both."""
from __future__ import annotations
import json, time, uuid, pathlib
from ..utils.config import ROOT

try:
    import mlflow  # noqa
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class Tracker:
    def __init__(self, experiment: str, use_mlflow: bool | None = None):
        self.experiment = experiment; self.use_mlflow = HAS_MLFLOW if use_mlflow is None else (use_mlflow and HAS_MLFLOW)
        self.dir = ROOT / "mlruns_local" / experiment; self.dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, name: str):
        return _Run(self, name)


class _Run:
    def __init__(self, tr: Tracker, name: str):
        self.tr, self.name, self.id = tr, name, uuid.uuid4().hex[:10]
        self.params, self.metrics, self.artifacts, self.t0 = {}, {}, [], time.time()
        self._ml = None

    def __enter__(self):
        if self.tr.use_mlflow:
            import mlflow
            mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}"); mlflow.set_experiment(self.tr.experiment); self._ml = mlflow.start_run(run_name=self.name)
        return self

    def log_params(self, d):
        self.params.update({k: str(v) for k, v in d.items()})
        if self._ml:
            import mlflow; mlflow.log_params({k: str(v)[:250] for k, v in d.items()})

    def log_metrics(self, d, step=None):
        flat = {k: float(v) for k, v in d.items() if isinstance(v, (int, float)) and v == v}
        self.metrics.update(flat)
        if self._ml:
            import mlflow; mlflow.log_metrics(flat, step=step)

    def log_artifact(self, p):
        self.artifacts.append(str(p))
        if self._ml:
            import mlflow; mlflow.log_artifact(str(p))

    def __exit__(self, *exc):
        rec = {"run_id": self.id, "name": self.name, "experiment": self.tr.experiment, "params": self.params, "metrics": self.metrics,
               "artifacts": self.artifacts, "duration_s": time.time() - self.t0, "status": "FAILED" if exc[0] else "FINISHED", "start": self.t0, "backend": "mlflow+local" if self._ml else "local"}
        (self.tr.dir / f"{self.id}.json").write_text(json.dumps(rec, indent=1))
        if self._ml:
            import mlflow; mlflow.end_run()


def list_runs(experiment: str | None = None):
    base = ROOT / "mlruns_local"; out = []
    if not base.exists(): return out
    for d in base.iterdir():
        if d.is_dir() and (experiment in (None, d.name)):
            out += [json.loads(f.read_text()) for f in d.glob("*.json")]
    return sorted(out, key=lambda r: -r["start"])


class _NullRun:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def log_params(self, d): pass
    def log_metrics(self, d, step=None): pass
    def log_artifact(self, p): pass


class NullTracker:
    """Used by tests / dry runs: records nothing."""
    def start_run(self, name): return _NullRun()
