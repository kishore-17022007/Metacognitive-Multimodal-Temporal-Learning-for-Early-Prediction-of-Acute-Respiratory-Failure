"""Survival evaluation with censoring. Harrell C-index (comparable pairs only) and a censoring-aware Brier at a fixed interval."""
import numpy as np


def concordance_index(time, event, risk) -> float | None:
    time, event, risk = map(np.asarray, (time, event, risk)); conc = ties = comp = 0.0
    idx = np.where(event == 1)[0]
    for i in idx:
        m = time > time[i]                      # pairs where i fails first and j is known to survive longer
        comp += m.sum(); conc += (risk[i] > risk[m]).sum(); ties += (risk[i] == risk[m]).sum()
    return None if comp == 0 else float((conc + 0.5 * ties) / comp)


def brier_at(time, event, surv_prob_at_k, k: int) -> dict:
    """Brier at interval k using only subjects whose status at k is known: event by k, or follow-up >= k.
    Simplification: no IPCW (documented). Subjects censored before k are dropped, which can bias toward earlier events."""
    time, event, s = map(np.asarray, (time, event, surv_prob_at_k))
    known = ((event == 1) & (time <= k)) | (time >= k)
    if known.sum() < 10: return {"brier": None, "n": int(known.sum()), "note": "insufficient known-status subjects"}
    alive = (time > k) | ((time == k) & (event == 0))
    return {"brier": float(np.mean((s[known] - alive[known].astype(float)) ** 2)), "n": int(known.sum()), "note": "no IPCW; censored-before-k excluded"}
