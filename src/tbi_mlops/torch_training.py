"""PyTorch training for the multimodal multi-task model + small deep ensemble. [UNTESTED in build sandbox: torch unavailable]
Run via: python scripts/train_advanced.py   (automatically used when torch is installed)."""
from __future__ import annotations
import time, json, numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from .data.dataset import prepare
from .models.multitask import MultiTaskModel, multitask_loss, survival_from_logits
from .evaluation.survival import concordance_index
from .evaluation.classification import curves
from .tracking.tracker import Tracker
from .utils.config import path
from .utils.reproducibility import set_seed, code_version


def _tensors(part, K):
    m = part["meta"]
    return TensorDataset(torch.tensor(part["X"]), torch.tensor(part["valid"]), torch.tensor(m.y.values), torch.tensor(m.label_known.values),
                         torch.tensor(m[[f"traj_{k}" for k in range(K)]].values, dtype=torch.float32), torch.tensor(m[[f"trajvalid_{k}" for k in range(K)]].values, dtype=torch.float32),
                         torch.tensor(m.surv_time.values, dtype=torch.long), torch.tensor(m.surv_event.values, dtype=torch.long))


def _batch(b): return {"y": b[2], "label_known": b[3], "traj": b[4], "traj_valid": b[5], "surv_time": b[6], "surv_event": b[7]}


@torch.no_grad()
def predict(model, ds, bs=256):
    model.eval(); P, S = [], []
    for b in DataLoader(ds, batch_size=bs):
        o = model(b[0], b[1]); P.append(torch.sigmoid(o["binary"]).numpy()); S.append(survival_from_logits(o["haz"]).numpy())
    return np.concatenate(P), np.concatenate(S)


def train_one(cfg, D, seed, tr_ds, va_ds, K, run=None):
    set_seed(seed); tc = cfg["training"]["torch"]; X = D["ds"]["X"]
    model = MultiTaskModel(X.shape[2], K, X.shape[1], cfg["model"]); opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    w = cfg["model"]["multitask"]; best, best_state, bad = 1e9, None, 0; ckpt = path("artifacts", "models", "torch", f"member_seed{seed}.pt")
    for ep in range(tc["epochs"]):
        model.train(); t0 = time.time(); tl = 0
        for b in DataLoader(tr_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=tc["num_workers"]):
            opt.zero_grad(); loss, _ = multitask_loss(model(b[0], b[1]), _batch(b), w); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += float(loss) * len(b[0])
        model.eval(); vl = 0
        with torch.no_grad():
            for b in DataLoader(va_ds, batch_size=256): vl += float(multitask_loss(model(b[0], b[1]), _batch(b), w)[0]) * len(b[0])
        vl /= len(va_ds); tl /= len(tr_ds)
        if run: run.log_metrics({"train_loss": tl, "val_loss": vl, "epoch_seconds": time.time() - t0}, step=ep)
        if vl < best - 1e-4: best, bad = vl, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}; torch.save(best_state, ckpt)   # state_dict only (weights_only-safe)
        else:
            bad += 1
            if bad >= tc["early_stopping_patience"]: break
    model.load_state_dict(best_state); return model, ep + 1, best


def train_torch_models(cfg, D=None):
    D = D or prepare(cfg); K = D["ds"]["K"]; n = min(cfg["model"]["ensemble"]["n_members"], 3); tr = Tracker("tbi-arf-torch")
    ds = {k: _tensors(v, K) for k, v in D["parts"].items()}; test = D["parts"]["test"]["meta"]; known = test.label_known.values; out = {}
    with tr.start_run("multitask-transformer-graph") as run:
        dataset_version = f"synthetic-{D['raw_hash']}"
        run.log_params({"seed": cfg["training"]["seed"], "n_members": n, "code_version": code_version(), "dataset_version": dataset_version, "data_provenance": "synthetic", "torch": torch.__version__, **{f"tf_{k}": v for k, v in cfg["model"]["transformer"].items()}})
        members = []
        for m in range(n):
            model, epochs, best = train_one(cfg, D, cfg["training"]["seed"] + m, ds["train"], ds["validation"], K, run); members.append(model); out[f"member{m}"] = {"epochs": epochs, "best_val_loss": best}
        P = np.stack([predict(m, ds["test"])[0] for m in members], 1); S = np.stack([predict(m, ds["test"])[1] for m in members], 1).mean(1); p = P.mean(1)
        y = test.y.values
        res = {"auroc": roc_auc_score(y[known], p[known]), "auprc": average_precision_score(y[known], p[known]), "ensemble_std_mean": float(P.std(1).mean()),
             "c_index": concordance_index(test.surv_time.values, test.surv_event.values, 1 - S[:, -1]), "curves": curves(y[known], p[known])}
        run.log_metrics({f"test_{k}": v for k, v in res.items() if v is not None}); out["test"] = res
    out["dataset_version"] = dataset_version
    out["seed"] = cfg["training"]["seed"]
    out["code_version"] = code_version()
    from .evaluation.reporting import write_json
    write_json(out, "torch_report.json")
    return out
