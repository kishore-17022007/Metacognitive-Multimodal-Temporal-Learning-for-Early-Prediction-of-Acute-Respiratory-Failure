import time, numpy as np
from collections import deque


class OpsMonitor:
    """Operational metrics: latency, requests, errors, status distribution."""
    def __init__(self, maxlen=2000):
        self.lat = deque(maxlen=maxlen); self.requests = 0; self.errors = 0; self.status = {}; self.started = time.time(); self.alerts = []

    def record(self, ms, status=None, error=False):
        self.requests += 1; self.errors += int(error)
        if not error: self.lat.append(ms)
        if status: self.status[status] = self.status.get(status, 0) + 1

    def summary(self, p95_alert_ms=250):
        a = np.array(self.lat) if self.lat else np.array([0.0])
        s = {"requests": self.requests, "errors": self.errors, "error_rate": self.errors / max(self.requests, 1), "p50_ms": float(np.percentile(a, 50)),
             "p95_ms": float(np.percentile(a, 95)), "status_distribution": dict(self.status), "uptime_s": time.time() - self.started}
        s["latency_alert"] = s["p95_ms"] > p95_alert_ms; return s
