import unittest
from helpers import *
from tbi_mlops.streaming.engine import StreamEngine
from tbi_mlops.streaming.replay import ONSET_OBS
from tbi_mlops.data.schema import *


def _eng(scn="normal", kind="nonevent", **kw):
    t = tiny(); g, q = long_patient(kind); ev = q.event_time_h if kind == "event" else None
    return StreamEngine(t["P"], g, scn, ev, reference=t["D"]["clean"], monitoring_cfg=t["cfg"]["monitoring"], **kw), g


def test_start_pause_resume_and_reset():
    e, g = _eng(); assert e.tick(10, 3600) == [] and e.i == 0, "not running -> no progress"
    e.start(); e.tick(1, 3600 * 3); n = e.i; assert n > 0 and e.running
    e.pause(); assert e.tick(10, 3600) == [] and e.i == n
    e.resume(); e.tick(1, 3600 * 3); assert e.i > n
    e.reset(); assert e.i == 0 and e.clock_h == 0 and not e.predictions and not e.running


def test_chronological_replay_and_prediction_updates():
    e, g = _eng(); e.start()
    for _ in range(12): e.step()
    h = [r["hours_since_admission"] for r in e.rows]; assert h == sorted(h) and h == list(g.sort_values(TIME_COL).hours_since_admission.values[:12])
    assert len(e.predictions) == 12 and [p["sim_hour"] for p in e.predictions] == h and all(p["inference_latency_ms"] > 0 for p in e.predictions)


def test_prediction_uses_only_past_observations():
    e, g = _eng(); [e.step() for _ in range(14)]; direct = tiny()["P"].predict_frame(pd.DataFrame(e.rows[:10])); direct.pop("_window")
    assert abs(direct["predicted_probability"] - e.predictions[9]["predicted_probability"]) < 1e-9


def test_event_logged_at_its_time_without_leakage():
    e, g = _eng(kind="event"); e.start(); guard = 0
    while e.running and guard < 10000: e.tick(1, 3600); guard += 1
    assert e.finished and not e.running
    ev = [x for x in e.events if x.kind == "event"]; assert len(ev) == 1 and abs(ev[0].sim_hour - e.event_time_h) < 1e-9
    assert max(r["hours_since_admission"] for r in e.rows) < e.event_time_h, "no observations at/after the event are replayed"


def test_abrupt_drift_alert_not_before_onset():
    e, g = _eng("abrupt_drift"); e.start()
    while not e.finished: e.tick(1, 3600)
    alerts = [a for a in e.alert_history if "input shift" in a["message"]]; assert alerts and e.drift["overall"] == "alert"
    onset_h = float(g.sort_values(TIME_COL).hours_since_admission.values[ONSET_OBS]); assert alerts[0]["sim_hour"] >= onset_h


def test_abnormal_inputs_flagged_not_reliable():
    e, g = _eng("abnormal_inputs"); e.start()
    while not e.finished: e.tick(1, 3600)
    post = [p for p in e.predictions[ONSET_OBS + 2:]]; assert np.mean([p["ood_status"]["flag"] for p in post]) > 0.7
    assert all(p["prediction_status"] != "reliable" for p in post if p["ood_status"]["flag"])


def test_high_latency_raises_alert_and_failures_counted():
    e, g = _eng("high_latency"); e.start(); [e.step() for _ in range(ONSET_OBS + 3)]
    assert any("latency" in a["message"] for a in e.alert_history) and e.ops.summary()["p95_ms"] > 250
    e2, _ = _eng(); e2.pred = None; e2.step(); assert e2.ops.errors == 1, "failed predictions are counted, not hidden"
