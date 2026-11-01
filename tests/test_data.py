from helpers import *
from tbi_mlops.data.schema import *
from tbi_mlops.data.validate_data import validate_clean, validate_raw_schema
from tbi_mlops.preprocessing.clean import clean_observations


def test_schema_and_synthetic_flag():
    t = tiny(); assert not validate_raw_schema(t["raw"]); assert (t["raw"].is_synthetic.astype(str) == "True").all()
    assert (t["raw"].data_source == "synthetic").all() and t["rep"]["synthetic"] is True


def test_generation_reproducible():
    cfg = tiny_cfg(); cfg["data"]["n_patients"] = 30
    a, pa, _ = generate(cfg["data"]); b, pb, _ = generate(cfg["data"])
    assert a.astype(str).equals(b.astype(str)) and pa.equals(pb)


def test_injected_errors_are_tracked_and_present():
    r = tiny()["rep"]
    for k in ("extra_missing_cells", "invalid_values_injected", "exact_duplicates_injected", "bad_timestamps_injected", "bad_dtype_cells_injected"): assert r[k] > 0, k
    assert r["n_rows_after_injection"] > r["n_rows_before_injection"]


def test_cleaning_fixes_and_reports():
    t = tiny(); before = t["raw"].astype(str).copy(); clean, rep = clean_observations(t["raw"])
    assert t["raw"].astype(str).equals(before), "original must be unchanged"
    assert not validate_clean(clean)
    inj = t["rep"]; assert rep["steps"]["duplicates_removed"] >= inj["exact_duplicates_injected"] * 0.9
    assert rep["steps"]["rows_removed_bad_timestamp"] >= 1 and rep["before"]["rows"] > rep["after"]["rows"]
    assert sum(rep["steps"]["unit_corrections"].values()) > 0 and "unresolved" in rep


def test_unit_normalisation_and_invalid_not_imputed():
    row = dict(patient_id="X", timestamp="2020-01-01T00:00:00", hours_since_admission=0.0, data_source="s", is_synthetic=True, **{f: np.nan for f in FEATURE_NAMES})
    row.update(spo2=0.97, fio2=40.0, paco2=5.3, hemoglobin=130.0, heart_rate=-10.0, resp_rate="N/A")
    c, rep = clean_observations(pd.DataFrame([row]))
    assert abs(c.spo2[0] - 97) < 1e-6 and abs(c.fio2[0] - 0.4) < 1e-6 and abs(c.paco2[0] - 39.75) < 1e-6 and abs(c.hemoglobin[0] - 13.0) < 1e-6
    assert np.isnan(c.heart_rate[0]) and np.isnan(c.resp_rate[0]), "invalid/non-numeric must become missing, not invented"


def test_chronological_and_duplicate_free_after_clean():
    c = tiny()["D"]["clean"]; assert not c.duplicated([ID_COL, TIME_COL]).any()
    assert all(g[TIME_COL].is_monotonic_increasing for _, g in c.groupby(ID_COL))


def test_outcome_and_censoring_consistency():
    t = tiny(); pat, m = t["pat"], t["D"]["ds"]["meta"]; ev = pat[pat.event_observed]
    assert (ev.event_time_h < ev.followup_end_h).all() and pat[~pat.event_observed].event_time_h.isna().all()
    K = t["D"]["ds"]["K"]; assert m.surv_time.between(1, K).all()
    assert (m[m.y == 1].surv_event == 1).all() and (m[m.surv_event == 1].y == 1).all()
    assert not m.patient_id.isin(pat[~pat.outcome_known].patient_id).any(), "unknown outcomes excluded"
    cens = m[(~m.label_known)]; assert (cens.surv_event == 0).all() and (cens.surv_time < K).all(), "censored windows: no binary label, censored survival"
