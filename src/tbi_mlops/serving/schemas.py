"""Pydantic request/response schemas. [UNTESTED in build sandbox: pydantic/fastapi unavailable]"""
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str
    timestamp: str = Field(description="ISO-8601 timestamp")
    hours_since_admission: Optional[float] = None
    gcs_total: Optional[float] = None; pupil_response: Optional[float] = None; sedated: Optional[float] = None
    resp_rate: Optional[float] = None; spo2: Optional[float] = None; o2_support: Optional[float] = None; fio2: Optional[float] = None
    pao2: Optional[float] = None; paco2: Optional[float] = None; heart_rate: Optional[float] = None
    sbp: Optional[float] = None; dbp: Optional[float] = None; map: Optional[float] = None
    lactate: Optional[float] = None; hemoglobin: Optional[float] = None


class PredictRequest(BaseModel):
    observations: list[Observation] = Field(min_length=1, description="Chronological history for ONE patient up to the prediction time")


class PredictResponse(BaseModel):
    prediction_status: str
    predicted_probability: float
    prediction_horizon_hours: float
    uncertainty_summary: dict
    ood_status: dict
    conformal_set: list[int]
    data_quality_flags: list[str]
    reliability_reasons: list[str]
    survival_curve: list[float]
    model_version: str
    prediction_time: str
    inference_latency_ms: float
    n_observations_used: int
    risk_above_threshold: bool
    disclaimer: str
