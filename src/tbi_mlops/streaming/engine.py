"""Streaming simulation engine (SYNTHETIC replay). Pure-Python state machine: the dashboard (or tests) call tick()/step()."""
from __future__ import annotations
import time
import numpy as np, pandas as pd
from ..data.schema import *
from .events import Event
from .replay import Replay, perturb, ONSET_OBS
from ..monitoring.drift import drift_report
from ..monitoring.latency import OpsMonitor


class StreamEngine:
    def __init__(self, predictor, patient_obs: pd.DataFrame, scenario="normal", event_time_h=None, reference: pd.DataFrame | None = None,
                 monitoring_cfg: dict | None = None, seed=0, drift_check_every=5, min_drift_obs=25):
        self.pred, self.scenario, self.event_time_h, self.ref, self.mcfg = predictor, scenario, event_time_h, reference, monitoring_cfg
        self.replay = Replay(patient_obs); self.seed = seed; self.drift_every, self.min_drift = drift_check_every, min_drift_obs
        self.reset()

    # ---- controls ----
    def reset(self):
        self.i = 0; self.clock_h = 0.0; self.running = False; self.rng = np.random.default_rng(self.seed); self.rows = []; self.predictions = []
        self.events: list[Event] = []; self.event_logged = False; self.ops = OpsMonitor(); self.alert_history = []; self.drift = None; self._drift_alerted = False
        self.events.append(Event(0.0, "control", f"reset (scenario={self.scenario}, SYNTHETIC)"))

    def start(self): self.running = True; self.events.append(Event(self.clock_h, "control", "start"))
    def pause(self): self.running = False; self.events.append(Event(self.clock_h, "control", "pause"))
    def resume(self): self.running = True; self.events.append(Event(self.clock_h, "control", "resume"))
    @property
    def finished(self): return self.i >= len(self.replay)

    # ---- advancing ----
    def step(self):
        """Process exactly the next observation (advance the clock to its timestamp)."""
        if self.finished:                     # no more observations: let the clock reach a pending event so it shows on the timeline
            if self.event_time_h is not None and not self.event_logged: self.clock_h = max(self.clock_h, self.event_time_h); self._maybe_event(self.clock_h)
            return None
        h = self.replay.hour(self.i); self.clock_h = max(self.clock_h, h); return self._process()

    def tick(self, wall_seconds: float, speed: float = 60.0):
        """Advance simulated clock by wall_seconds*speed seconds and process all observations now due. Only when running."""
        if not self.running: return []
        self.clock_h += wall_seconds * speed / 3600.0; out = []
        while not self.finished and self.replay.hour(self.i) <= self.clock_h + 1e-9: out.append(self._process())
        self._maybe_event(self.clock_h)
        if self.finished and (self.event_time_h is None or self.event_logged): self.running = False   # keep clock running until a pending event is shown
        return out

    def _maybe_event(self, h):
        if self.event_time_h is not None and not self.event_logged and h >= self.event_time_h:
            self.event_logged = True; self.events.append(Event(float(self.event_time_h), "event", "Simulated ARF event (synthetic label) occurs", {}))

    def _process(self):
        i = self.i; row = self.replay.row(i); h = self.replay.hour(i); self.i += 1
        k = i - ONSET_OBS if i >= ONSET_OBS else -1
        row, extra_ms = perturb(row, self.scenario, k, self.rng); self.rows.append(row)
        self._maybe_event(h)       # event marker placed at its own time, after observations preceding it
        self.events.append(Event(h, "observation", "observation received"))
        hist = pd.DataFrame(self.rows)       # ONLY observations received so far -> no future leakage
        try:
            res = self.pred.predict_frame(hist); res.pop("_window", None); res["inference_latency_ms"] += extra_ms
            self.ops.record(res["inference_latency_ms"], res["prediction_status"]); res["sim_hour"] = h; self.predictions.append(res)
            self.events.append(Event(h, "prediction", f"risk={res['predicted_probability']:.3f} status={res['prediction_status']}", {"status": res["prediction_status"]}))
            if res["prediction_status"] in ("out_of_distribution", "review_required"): self._alert(h, f"reliability: {res['prediction_status']}")
            if self.mcfg and res["inference_latency_ms"] > self.mcfg["latency"]["p95_alert_ms"]: self._alert(h, f"latency {res['inference_latency_ms']:.0f} ms above threshold")
        except Exception as e:      # failed predictions are counted, not hidden
            self.ops.record(0, error=True); self._alert(h, f"prediction failed: {type(e).__name__}")
        self._drift_check(h)
        return self.predictions[-1] if self.predictions else None

    def _alert(self, h, msg):
        a = {"sim_hour": h, "message": msg}; self.alert_history.append(a); self.events.append(Event(h, "alert", msg))

    def _drift_check(self, h):
        """Within-stream input-shift monitor: recent-window mean vs the stream's own early baseline, in reference-population SD units.
        Real physiological deterioration can also trigger it; it flags *input change*, not its cause."""
        if self.ref is None or self.mcfg is None: return
        sc = self.mcfg["stream_shift"]; n = len(self.rows)
        if n < sc["baseline_obs"] + sc["recent_obs"] or n % self.drift_every: return
        cur = pd.DataFrame(self.rows); base, rec = cur.iloc[: sc["baseline_obs"]], cur.iloc[-sc["recent_obs"]:]
        shifted = []
        for f in FEATURES:
            if not f.continuous: continue
            sd = float(self.ref[f.name].std()) or 1.0
            b, r = base[f.name].mean(), rec[f.name].mean()
            miss_shift = abs(rec[f.name].isna().mean() - base[f.name].isna().mean())
            if (b == b and r == r and abs(r - b) / sd >= sc["z_alert"]) or miss_shift >= 0.5: shifted.append(f.name)
        self.drift = {"method": "stream_shift", "shifted_features": shifted, "overall": "alert" if len(shifted) >= sc["min_features"] else "ok"}
        if self.drift["overall"] == "alert" and not getattr(self, "_drift_alerted", False):
            self._drift_alerted = True; self._alert(h, f"input shift alert: {', '.join(shifted[:5])}")
