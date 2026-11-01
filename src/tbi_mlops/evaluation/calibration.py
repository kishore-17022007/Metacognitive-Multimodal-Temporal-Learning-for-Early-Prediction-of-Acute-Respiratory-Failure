import numpy as np


def calibration_curve_data(y, p, bins: int = 10):
    """Equal-width bins on [0,1]; returns (mean_pred, frac_pos, counts)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1); mp, fp, ct = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum(): mp.append(p[m].mean()); fp.append(y[m].mean()); ct.append(int(m.sum()))
    return np.array(mp), np.array(fp), np.array(ct)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    mp, fp, ct = calibration_curve_data(y, p, bins)
    return float((ct / ct.sum() * np.abs(mp - fp)).sum()) if len(ct) else float("nan")
