"""Internal observation schema. Single source of truth for features, groups, units and plausible ranges."""
from dataclasses import dataclass

ID_COL, TIME_COL, HOURS_COL = "patient_id", "timestamp", "hours_since_admission"


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    unit: str
    lo: float          # plausible range (outside -> invalid -> NaN)
    hi: float
    continuous: bool = True


FEATURES = [
    Feature("gcs_total", "neurological", "points", 3, 15, False),
    Feature("pupil_response", "neurological", "0=none,1=sluggish,2=reactive", 0, 2, False),
    Feature("sedated", "neurological", "0/1", 0, 1, False),
    Feature("resp_rate", "respiratory", "breaths/min", 4, 60),
    Feature("spo2", "respiratory", "%", 50, 100),
    Feature("o2_support", "respiratory", "0=RA,1=nasal,2=mask/HFNC,3=NIV", 0, 3, False),
    Feature("fio2", "respiratory", "fraction", 0.21, 1.0),
    Feature("pao2", "respiratory", "mmHg", 20, 600),
    Feature("paco2", "respiratory", "mmHg", 10, 150),
    Feature("heart_rate", "cardiovascular", "bpm", 20, 250),
    Feature("sbp", "cardiovascular", "mmHg", 40, 260),
    Feature("dbp", "cardiovascular", "mmHg", 20, 160),
    Feature("map", "cardiovascular", "mmHg", 30, 180),
    Feature("lactate", "laboratory", "mmol/L", 0.2, 30),
    Feature("hemoglobin", "laboratory", "g/dL", 3, 22),
]
FEATURE_NAMES = [f.name for f in FEATURES]
GROUPS = ["neurological", "respiratory", "cardiovascular", "laboratory"]
GROUP_OF = {f.name: f.group for f in FEATURES}
N_F = len(FEATURES)
META_COLS = ["data_source", "is_synthetic"]
OBS_COLUMNS = [ID_COL, TIME_COL, HOURS_COL] + FEATURE_NAMES + META_COLS
PATIENT_COLUMNS = ["patient_id", "followup_end_h", "event_observed", "event_time_h", "outcome_known", "data_source", "is_synthetic"]

# Unit-normalisation rules (applied only where the value is implausible in the canonical unit but plausible in the alternate).
UNIT_RULES = {
    "spo2": "fraction (0-1] -> %  (x100)",
    "fio2": "percent (1,100] -> fraction (/100)",
    "paco2": "kPa (0,15) -> mmHg (x7.5)",
    "hemoglobin": "g/L (>30) -> g/dL (/10)",
}

OUTCOME_DEFINITION = (
    "RESEARCH LABEL (synthetic): acute respiratory failure (ARF) event time is drawn from a configurable generator; "
    "physiology drifts toward worse values before the event for non-abrupt events. This is NOT a validated clinical "
    "criterion and must be replaced by a clinically adjudicated definition for real data. Oxygen use, ventilation or a "
    "single abnormal vital sign alone do not define ARF."
)
