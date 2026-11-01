from helpers import *
from tbi_mlops.monitoring.drift import drift_report, psi
from tbi_mlops.monitoring.data_quality import batch_quality
from tbi_mlops.monitoring.performance import delayed_performance, matured_mask
from tbi_mlops.monitoring.latency import OpsMonitor
from tbi_mlops.data.schema import *


def test_drift_detects_shift_not_noise():
    c = tiny()["D"]["clean"]; mc = tiny()["cfg"]["monitoring"]["drift"]; h = len(c) // 2; same = drift_report(c.iloc[:h], c.iloc[h:].sample(frac=1, random_state=0), mc)
    shifted = c.iloc[h:].copy(); shifted["heart_rate"] += 40; shifted.loc[shifted.sample(frac=.5, random_state=1).index, "spo2"] = np.nan
    r = drift_report(c.iloc[:h], shifted, mc); lv = {f["feature"]: f["level"] for f in r["features"]}
    assert lv["heart_rate"] == "alert" and lv["spo2"] == "alert" and r["overall"] == "alert"; assert psi(np.random.default_rng(0).normal(size=500), np.random.default_rng(1).normal(size=500)) < 0.1


def test_batch_data_quality_flags():
    t = tiny(); q = batch_quality(t["raw"], t["cfg"]["monitoring"]); assert q["unparseable_timestamps"] > 0 and "timestamp_anomalies" in q["flags"] and q["duplicate_rate"] > 0
    assert "schema_violation" in batch_quality(t["raw"].drop(columns=["spo2"]), t["cfg"]["monitoring"])["flags"]


def test_delayed_performance_ignores_unmatured():
    meta = pd.DataFrame({"t_index_h": np.arange(200, dtype=float), "y": np.tile([0, 0, 0, 1], 50), "label_known": True}); p = meta.y.values * 0.8 + 0.1
    early = delayed_performance(meta, p, 0.5, 6, 20, tiny()["cfg"]["monitoring"]); assert early["status"] == "insufficient_matured_labels" and early["n_pending"] > 150
    late = delayed_performance(meta, p, 0.5, 6, 150, tiny()["cfg"]["monitoring"]); assert late["status"] == "ok" and late["n_matured"] == int(matured_mask(meta.t_index_h.values, 6, 150).sum()) and late["auroc"] == 1.0


def test_ops_monitor():
    m = OpsMonitor(); [m.record(10, "reliable") for _ in range(9)]; m.record(0, error=True); m.record(900, "review_required"); s = m.summary(250)
    assert s["requests"] == 11 and s["errors"] == 1 and s["status_distribution"]["reliable"] == 9 and s["p95_ms"] > 250 and s["latency_alert"]
