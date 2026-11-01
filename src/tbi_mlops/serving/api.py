"""FastAPI service. [UNTESTED in build sandbox: fastapi/uvicorn unavailable; Predictor logic underneath IS tested]
Run: uvicorn src.tbi_mlops.serving.api:app --reload      Research prototype - NOT a validated clinical service."""
from __future__ import annotations
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from .predictor import Predictor, ModelNotAvailable, InputError, DISCLAIMER
from .schemas import PredictRequest, PredictResponse
from ..monitoring.latency import OpsMonitor
from ..tracking import registry
from ..utils.config import load_config
from ..utils.logging import get_logger

log = get_logger("tbi_api"); app = FastAPI(title="TBI ARF early-warning research prototype", description=DISCLAIMER, version="0.1.0")
CFG = load_config(); OPS = OpsMonitor(); _P: dict = {}


def predictor() -> Predictor:
    if "p" not in _P: _P["p"] = Predictor(cfg=CFG)          # loaded lazily; missing artifacts -> 503, not a crash
    return _P["p"]


@app.exception_handler(Exception)
async def unhandled(_, exc):                                  # never leak stack traces to clients
    log.error("unhandled error", extra={"fields": {"type": type(exc).__name__}}); OPS.record(0, error=True)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


@app.get("/health")
def health():
    try: predictor(); ok = True
    except ModelNotAvailable: ok = False
    return {"status": "ok" if ok else "degraded", "model_loaded": ok, "disclaimer": DISCLAIMER}


@app.get("/model-info")
def model_info():
    try: p = predictor()
    except ModelNotAvailable as e: raise HTTPException(503, str(e))
    return {"version": p.version, "metadata": {k: p.meta[k] for k in ("created", "data_provenance", "dataset_version", "seed", "code_version", "profile", "disclaimer")}, "manifest": registry.manifest()["versions"].get(p.version, {}).get("status")}


def _run(records):
    t0 = time.perf_counter()
    try:
        out = predictor().predict_records(records)
    except ModelNotAvailable as e: OPS.record(0, error=True); raise HTTPException(503, str(e))
    except InputError as e: OPS.record(0, error=True); raise HTTPException(422, str(e))
    OPS.record((time.perf_counter() - t0) * 1000, out["prediction_status"]); log.info("prediction", extra={"fields": {"status": out["prediction_status"], "latency_ms": out["inference_latency_ms"]}})   # no patient data logged
    return out


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest): return _run([o.model_dump(exclude_none=True) for o in req.observations])


@app.post("/predict-window", response_model=PredictResponse)
def predict_window(req: PredictRequest): return _run([o.model_dump(exclude_none=True) for o in req.observations][-CFG["data"]["outcome"]["window_length"]:])


@app.get("/metrics")
def metrics(): return OPS.summary(CFG["monitoring"]["latency"]["p95_alert_ms"])


@app.get("/monitoring/summary")
def monitoring_summary():
    return {"operational": OPS.summary(CFG["monitoring"]["latency"]["p95_alert_ms"]), "model_version": _P["p"].version if "p" in _P else None}
