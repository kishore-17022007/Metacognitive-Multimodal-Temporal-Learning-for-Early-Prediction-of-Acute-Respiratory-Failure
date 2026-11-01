"""Reliability assessment object. Transparent, configurable rules; NO validated universal trust score.
Priority of status: insufficient_data > out_of_distribution > review_required > reliable."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field, asdict


@dataclass
class Reliability:
    predicted_probability: float
    raw_probability: float
    ensemble_std: float
    ensemble_agreement: float
    calibration_status: str
    ood_score: float
    ood_flag: bool
    conformal_set: list
    input_quality_flags: list
    status: str
    reasons: list
    model_version: str
    prediction_time: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    experimental_score: float | None = None
    def to_dict(self): return asdict(self)


def decide(p_cal, p_raw, std, agree, cal_status, ood_score, ood_flag, cset, quality_flags, rcfg, model_version) -> Reliability:
    reasons = []
    if "insufficient_history" in quality_flags: reasons.append("fewer observations than needed")
    if ood_flag: reasons.append("input unfamiliar vs training distribution (OOD score above validation-derived threshold)")
    if std > rcfg["max_ensemble_std"]: reasons.append(f"ensemble disagreement std={std:.3f} > {rcfg['max_ensemble_std']}")
    if agree < rcfg["min_agreement"]: reasons.append(f"member agreement {agree:.2f} < {rcfg['min_agreement']}")
    if len(cset) != 1: reasons.append(f"conformal set is {'empty' if not cset else 'ambiguous {0,1}'}")
    lo, hi = rcfg["review_probability_band"]
    if lo <= p_cal <= hi: reasons.append("probability within borderline band")
    if "high_missingness" in quality_flags: reasons.append("high missingness in window")
    if "insufficient_history" in quality_flags: status = "insufficient_data"
    elif ood_flag: status = "out_of_distribution"
    elif reasons: status = "review_required"
    else: status = "reliable"
    # EXPERIMENTAL (documented, unvalidated): 1 - min(1, std/max_std) * 0.5 - 0.5*ood_flag, clipped to [0,1]
    exp = max(0.0, min(1.0, 1 - 0.5 * min(1.0, std / rcfg["max_ensemble_std"]) - 0.5 * float(ood_flag)))
    return Reliability(float(p_cal), float(p_raw), float(std), float(agree), cal_status, float(ood_score), bool(ood_flag), list(cset), list(quality_flags), status, reasons, model_version, experimental_score=exp)
