"""Synthetic ICU generator (SYNTHETIC ONLY). Run: python -m src.tbi_mlops.data.generate_synthetic [--profile dev]"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
from .schema import *
from ..utils.config import load_config, path


def _patient_series(rng, pid, admit, cfg, is_event, abrupt, scale):
    stay = rng.uniform(cfg["stay_hours_min"], cfg["stay_hours_max"])
    T = rng.uniform(18, stay - 8) if is_event else np.nan
    end = T if is_event else stay                    # observations stop at the event (patient escalated)
    t = [0.0]
    while True:
        nt = t[-1] + max(0.25, rng.normal(1.5 * scale, 0.5 * scale))
        if nt >= end:
            break
        t.append(nt)
    t = np.array(t); n = len(t)
    base = rng.normal(0, 0.6)
    s = base + np.cumsum(rng.normal(0, 0.08, n))     # slow random walk
    if is_event and not abrupt:                       # prodrome: severity ramps up before T
        tau = rng.uniform(10, 24)
        s += 3.0 * np.clip((t - (T - tau)) / tau, 0, 1)
    if (not is_event) and rng.random() < 0.25:        # hard negatives: transient bump that resolves
        c = rng.uniform(10, max(11, end - 5)); s += 1.8 * np.exp(-((t - c) ** 2) / (2 * 3.0 ** 2))
    N = lambda sd: rng.normal(0, sd, n)
    sedated = (rng.random() < 0.4) * np.ones(n)
    gcs = np.clip(np.round(10 + N(1.5) - 0.3 * s), 3, 15)
    gcs = np.where((sedated == 1) & (rng.random(n) < 0.5), np.nan, gcs)
    pupil = rng.choice([0, 1, 2], n, p=[0.03, 0.12, 0.85]).astype(float)
    sup = np.digitize(s, [0.8, 1.8, 2.8]).astype(float)
    fio2 = np.clip(0.21 + 0.14 * sup + np.abs(N(0.03)), 0.21, 1.0)
    sbp = 125 - 3 * s + N(8); dbp = 75 - 1.5 * s + N(5)
    lab = rng.random(n) < 0.2                         # labs measured less often than vitals
    labv = lambda x: np.where(lab, x, np.nan)
    df = pd.DataFrame({
        ID_COL: pid, HOURS_COL: t, "gcs_total": gcs, "pupil_response": pupil, "sedated": sedated,
        "resp_rate": 16 + 3 * s + N(2), "spo2": np.minimum(100, 97 - 1.5 * s + N(1)), "o2_support": sup, "fio2": fio2,
        "pao2": labv(95 - 12 * s + N(8)), "paco2": labv(40 + 2 * s + N(3)),
        "heart_rate": 80 + 5 * s + N(6), "sbp": sbp, "dbp": dbp, "map": (sbp + 2 * dbp) / 3,
        "lactate": labv(1.2 + 0.4 * s + np.abs(N(0.4))), "hemoglobin": labv(12.5 - 0.1 * s + N(1.2)),
    })
    df[TIME_COL] = (admit + pd.to_timedelta(t, unit="h")).dt.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(admit, "dt") else (pd.Timestamp(admit) + pd.to_timedelta(t, unit="h")).strftime("%Y-%m-%dT%H:%M:%S")
    return df, dict(patient_id=pid, followup_end_h=float(stay), event_observed=bool(is_event), event_time_h=float(T) if is_event else np.nan)


def generate(cfg: dict, seed: int | None = None):
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    pc = cfg
    frames, pats = [], []
    for i in range(pc["n_patients"]):
        is_event = rng.random() < pc["event_fraction"]
        abrupt = is_event and rng.random() < pc["abrupt_event_fraction"]
        admit = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=float(rng.uniform(0, 24 * 700)))
        d, p = _patient_series(rng, f"P{i:05d}", admit, pc, is_event, abrupt, rng.choice([1.0, 1.0, 2.0, 3.0]))
        frames.append(d); pats.append(p)
    obs = pd.concat(frames, ignore_index=True)
    pat = pd.DataFrame(pats)
    pat["outcome_known"] = rng.random(len(pat)) >= pc["unknown_outcome_fraction"]
    for df in (obs, pat):
        df["data_source"] = "synthetic"; df["is_synthetic"] = True
    clean_truth_rows = len(obs)
    obs, rep = inject_errors(obs, pc["injection"], rng)
    rep.update(n_patients=len(pat), n_rows_before_injection=clean_truth_rows, n_rows_after_injection=len(obs),
               event_rate_patients=float(pat.event_observed.mean()), unknown_outcome_patients=int((~pat.outcome_known).sum()),
               outcome_definition=OUTCOME_DEFINITION, synthetic=True)
    return obs[OBS_COLUMNS], pat[PATIENT_COLUMNS], rep


def inject_errors(obs: pd.DataFrame, inj: dict, rng):
    obs = obs.copy(); n = len(obs); rep = {}
    cont = [f.name for f in FEATURES]
    for c in cont:
        obs[c] = obs[c].astype(object)
    def cells(rate):
        k = int(round(rate * n * len(cont)))
        return rng.integers(0, n, k), rng.choice(cont, k)
    # extra missingness (vitals only)
    vit = [c for c in cont if GROUP_OF[c] in ("respiratory", "cardiovascular") and c not in ("pao2", "paco2")]
    k = int(inj["extra_missing_rate"] * n * len(vit)); r, c = rng.integers(0, n, k), rng.choice(vit, k)
    for a, b in zip(r, c): obs.iat[a, obs.columns.get_loc(b)] = np.nan
    rep["extra_missing_cells"] = k
    # unit errors
    cnt = {"spo2": 0, "fio2": 0, "paco2": 0, "hemoglobin": 0}
    for a in rng.choice(n, int(inj["unit_error_rate"] * n), replace=False):
        col = rng.choice(list(cnt)); j = obs.columns.get_loc(col); v = obs.iat[a, j]
        if v is None or (isinstance(v, float) and np.isnan(v)): continue
        v = float(v)
        obs.iat[a, j] = {"spo2": v / 100, "fio2": v * 100, "paco2": v / 7.5, "hemoglobin": v * 10}[col]; cnt[col] += 1
    rep["unit_errors_injected"] = cnt
    # invalid values
    bad = {"spo2": [0, 150, -5], "heart_rate": [-10, 400], "resp_rate": [-3, 200], "sbp": [0, 500], "gcs_total": [0, 20]}
    k = int(inj["invalid_rate"] * n)
    for a in rng.choice(n, k, replace=False):
        col = rng.choice(list(bad)); obs.iat[a, obs.columns.get_loc(col)] = float(rng.choice(bad[col]))
    rep["invalid_values_injected"] = k
    # plausible-but-extreme outliers
    ext = {"lactate": 25.0, "resp_rate": 55.0, "heart_rate": 210.0, "sbp": 230.0}
    k = int(inj["outlier_rate"] * n)
    for a in rng.choice(n, k, replace=False):
        col = rng.choice(list(ext)); obs.iat[a, obs.columns.get_loc(col)] = ext[col]
    rep["outliers_injected"] = k
    # wrong dtype / corrupted tokens
    for key, toks in (("bad_dtype", ["N/A", "?", "null", "missing"]), ("corrupted", ["###", "1e999x", "--"])):
        k = int(inj[key + "_rate"] * n * len(cont)); r, c = rng.integers(0, n, k), rng.choice(cont, k)
        for a, b in zip(r, c): obs.iat[a, obs.columns.get_loc(b)] = str(rng.choice(toks))
        rep[key + "_cells_injected"] = k
    # bad timestamps
    idx = rng.choice(n, int(inj["bad_timestamp_rate"] * n), replace=False)
    obs.loc[obs.index[idx], TIME_COL] = "not-a-time"; rep["bad_timestamps_injected"] = len(idx)
    # duplicates (exact + conflicting)
    dup = obs.iloc[rng.choice(n, int(inj["duplicate_rate"] * n), replace=False)].copy()
    cdup = obs.iloc[rng.choice(n, int(inj["conflicting_duplicate_rate"] * n), replace=False)].copy()
    cdup["heart_rate"] = [float(x) + 7 if isinstance(x, (int, float)) and not pd.isna(x) else x for x in cdup["heart_rate"]]
    rep["exact_duplicates_injected"], rep["conflicting_duplicates_injected"] = len(dup), len(cdup)
    obs = pd.concat([obs, dup, cdup], ignore_index=True)
    obs = obs.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)  # unsorted on disk
    return obs, rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", default=None); ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.profile); obs, pat, rep = generate(cfg["data"], a.seed)
    out = path("data", "synthetic", "raw_observations.csv"); obs.to_csv(out, index=False)
    pat.to_csv(path("data", "synthetic", "patients.csv"), index=False)
    path("artifacts", "reports", "x").parent.joinpath("synthetic_generation_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[synthetic] wrote {len(obs)} rows / {len(pat)} patients -> {out}  (SYNTHETIC DATA)")


if __name__ == "__main__":
    main()
