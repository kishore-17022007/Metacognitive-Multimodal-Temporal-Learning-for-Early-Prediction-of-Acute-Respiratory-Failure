import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve
from .calibration import expected_calibration_error


def pick_threshold(y, p) -> float:
    """Validation-selected threshold maximising F1."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return float(th[np.nanargmax(f1[:-1])]) if len(th) else 0.5


def classification_metrics(y, p, threshold: float, ece_bins: int = 10) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, float); yhat = (p >= threshold).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    both = len(np.unique(y)) == 2
    return {"n": int(len(y)), "n_positive": int(y.sum()), "threshold": float(threshold),
            "auroc": float(roc_auc_score(y, p)) if both else None, "auprc": float(average_precision_score(y, p)) if both else None,
            "precision": float(pr), "recall": float(rc), "f1": float(f1), "brier": float(brier_score_loss(y, p)),
            "ece": expected_calibration_error(y, p, ece_bins), "confusion_matrix": cm.tolist()}


def curves(y, p):
    fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "precision": pr.tolist(), "recall": rc.tolist()}


def latency_ms(fn, X, repeats: int = 5) -> dict:
    """Per-window latency of fn on a single window (median / p95)."""
    ts = []
    for i in range(min(len(X), 50)):
        for _ in range(1):
            t = time.perf_counter(); fn(X[i:i + 1]); ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95))}
