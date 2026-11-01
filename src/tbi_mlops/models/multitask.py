"""Fusion + three heads + multi-task loss. [UNTESTED in build sandbox: torch unavailable]
Heads: (1) binary ARF-within-horizon logit, (2) trajectory: P(event by end of interval k) for k=1..K (cumulative),
(3) discrete-time survival hazards h_k. Survival NLL handles censoring explicitly (see survival_nll)."""
import torch, torch.nn as nn, torch.nn.functional as Fn
from .temporal_transformer import TemporalTransformer
from .physiological_graph import PhysiologicalGraph


def survival_nll(logits, time, event):
    """logits (B,K) hazards; time in 1..K (event interval or last observed interval); event 1/0.
    event=1: -sum_{k<t} log(1-h_k) - log h_t.   event=0 (censored after surviving interval t): -sum_{k<=t} log(1-h_k)."""
    B, K = logits.shape; k = torch.arange(1, K + 1, device=logits.device)[None]; t = time[:, None]
    log_h, log_s = Fn.logsigmoid(logits), Fn.logsigmoid(-logits)
    surv_part = ((k < t) | ((k == t) & (event[:, None] == 0))).float() * log_s
    ev_part = ((k == t) & (event[:, None] == 1)).float() * log_h
    return -(surv_part + ev_part).sum(1).mean()


def survival_from_logits(logits):
    return torch.cumprod(1 - torch.sigmoid(logits), dim=1)       # monotone non-increasing


class MultiTaskModel(nn.Module):
    def __init__(self, in_dim, K, window_length, mcfg: dict):
        super().__init__(); t, g = mcfg["transformer"], mcfg["graph"]; d = t["d_model"]
        self.temporal = TemporalTransformer(in_dim, window_length, d, t["n_heads"], t["n_layers"], t["dim_ff"], t["dropout"])
        self.graph = PhysiologicalGraph(g["node_dim"], g["n_heads"], g["n_layers"], g["dropout"], out_dim=d)
        self.gate = nn.Linear(2 * d, d)                          # gated fusion: h = g*h_t + (1-g)*h_g
        self.binary = nn.Linear(d, 1); self.traj = nn.Linear(d, K); self.haz = nn.Linear(d, K)

    def forward(self, x, valid):
        ht, hg = self.temporal(x, valid), self.graph(x, valid); gate = torch.sigmoid(self.gate(torch.cat([ht, hg], 1))); h = gate * ht + (1 - gate) * hg
        return {"binary": self.binary(h).squeeze(-1), "traj": self.traj(h), "haz": self.haz(h), "gate": gate.detach()}


def multitask_loss(out, batch, w):
    """batch: y, label_known, traj, traj_valid, surv_time, surv_event."""
    lk = batch["label_known"].float(); bce = Fn.binary_cross_entropy_with_logits(out["binary"], batch["y"].float(), reduction="none")
    lb = (bce * lk).sum() / lk.sum().clamp(min=1)
    lt = (Fn.binary_cross_entropy_with_logits(out["traj"], batch["traj"], reduction="none") * batch["traj_valid"]).sum() / batch["traj_valid"].sum().clamp(min=1)
    ls = survival_nll(out["haz"], batch["surv_time"], batch["surv_event"])
    return w["w_binary"] * lb + w["w_trajectory"] * lt + w["w_survival"] * ls, {"binary": float(lb), "trajectory": float(lt), "survival": float(ls)}
