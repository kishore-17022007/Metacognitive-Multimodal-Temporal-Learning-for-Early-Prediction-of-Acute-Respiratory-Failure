import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .calibration import calibration_curve_data
from ..utils.config import path


def _save(fig, name):
    p = path("artifacts", "graphs", name); p.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig); return str(p)


def plot_roc_pr(curves: dict):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for n, c in curves.items():
        ax[0].plot(c["fpr"], c["tpr"], label=n); ax[1].plot(c["recall"], c["precision"], label=n)
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR"); ax[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    ax[0].legend(fontsize=7); return _save(fig, "roc_pr.png")


def plot_calibration(y, raw, cal, bins=10):
    fig, ax = plt.subplots(figsize=(4.5, 4)); ax.plot([0, 1], [0, 1], "k:")
    for n, p in (("raw ensemble", raw), ("calibrated", cal)):
        mp, fp, _ = calibration_curve_data(y, p, bins); ax.plot(mp, fp, "o-", label=n)
    ax.set(title=f"Calibration (test, {bins} equal-width bins)", xlabel="mean predicted", ylabel="observed fraction"); ax.legend(); return _save(fig, "calibration.png")


def plot_temporal(att, valid, name="temporal_importance.png", title="Step attribution (occlusion)"):
    att, valid = np.asarray(att, dtype=float), np.asarray(valid, dtype=bool)
    fig, ax = plt.subplots(figsize=(6, 3)); idx = np.arange(len(att)) - len(att) + 1
    ax.bar(idx[valid], att[valid]); ax.set(title=title, xlabel="steps before prediction index (0 = latest)", ylabel="risk change when step removed"); return _save(fig, name)


def plot_model_metrics(metrics, name="model_metrics.png"):
    frame = pd.DataFrame(metrics).T
    selected = [column for column in ("auroc", "auprc", "f1") if column in frame]
    ax = frame[selected].plot(kind="bar", figsize=(11, 5), ylim=(0, 1), rot=25)
    ax.set(title="Model metrics on held-out synthetic test data", ylabel="score", xlabel="model")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.2)
    return _save(ax.figure, name)


def plot_global_importance(features, name="global_feature_importance.png"):
    ordered = list(reversed(features[:15])); fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([row["name"] for row in ordered], [row["importance"] for row in ordered], xerr=[row.get("std", 0) for row in ordered])
    ax.set(title="Global permutation importance (validation data)", xlabel="change in average precision")
    return _save(fig, name)


def plot_group_importance(group_importance, name="feature_group_importance.png"):
    fig, ax = plt.subplots(figsize=(8, 4)); names = list(group_importance); values = [group_importance[key] for key in names]
    ax.bar(names, values); ax.set(title="Feature-group occlusion contribution", ylabel="risk change")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, name)


def plot_temporal_heatmap(values, feature_names, relative_steps, name, title, colorbar_label):
    matrix = np.asarray(values, dtype=float).T
    width = max(8, min(16, len(relative_steps) * 0.65))
    fig, ax = plt.subplots(figsize=(width, max(5, len(feature_names) * 0.34)))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="RdBu_r")
    ax.set(yticks=np.arange(len(feature_names)), yticklabels=feature_names, xticks=np.arange(len(relative_steps)), xticklabels=relative_steps,
           title=title, xlabel="steps before prediction index (0 = latest)")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    return _save(fig, name)


def write_json(obj, name):
    p = path("artifacts", "reports", name); p.write_text(json.dumps(obj, indent=2, default=str)); return str(p)
