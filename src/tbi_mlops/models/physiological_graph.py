"""Physiological feature-group graph with attention message passing. [UNTESTED in build sandbox: torch unavailable]
Nodes: neurological, respiratory, cardiovascular, laboratory, temporal_context.
Edges (documented, undirected, + self loops): neuro-resp, neuro-cardio, resp-cardio, resp-lab, cardio-lab, context-all.
These are *modelling priors for information exchange*, NOT causal claims. Learned attention is descriptive only."""
import torch, torch.nn as nn, torch.nn.functional as Fn
from ..data.schema import FEATURES, GROUPS, N_F

NODES = GROUPS + ["temporal_context"]
EDGES = [("neurological", "respiratory"), ("neurological", "cardiovascular"), ("respiratory", "cardiovascular"), ("respiratory", "laboratory"), ("cardiovascular", "laboratory")] + [("temporal_context", g) for g in GROUPS]


def adjacency() -> torch.Tensor:
    n = len(NODES); A = torch.eye(n)
    for a, b in EDGES: i, j = NODES.index(a), NODES.index(b); A[i, j] = A[j, i] = 1
    return A


def node_channel_indices():
    """channel indices (value, observed, age) per node; context node uses time channels."""
    idx = {g: [] for g in GROUPS}
    for j, f in enumerate(FEATURES): idx[f.group] += [j, N_F + j, 2 * N_F + j]
    idx["temporal_context"] = [3 * N_F, 3 * N_F + 1]
    return [idx[n] for n in NODES]


class GATLayer(nn.Module):
    def __init__(self, dim, heads=2, dropout=0.1):
        super().__init__(); assert dim % heads == 0
        self.h, self.dk = heads, dim // heads; self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dk) * 0.1); self.a_dst = nn.Parameter(torch.randn(heads, self.dk) * 0.1); self.drop = nn.Dropout(dropout)

    def forward(self, h, A):
        B, N, D = h.shape; z = self.W(h).view(B, N, self.h, self.dk)
        e = (z * self.a_src).sum(-1)[:, :, None, :] + (z * self.a_dst).sum(-1)[:, None, :, :]        # (B,N,N,H)
        e = Fn.leaky_relu(e, 0.2).masked_fill(A[None, :, :, None] == 0, float("-inf"))
        att = self.drop(torch.softmax(e, dim=2))
        out = torch.einsum("bijh,bjhd->bihd", att, z).reshape(B, N, D)
        return Fn.elu(out) + h, att.detach()


class PhysiologicalGraph(nn.Module):
    def __init__(self, node_dim=16, heads=2, layers=2, dropout=0.1, out_dim=32):
        super().__init__(); self.register_buffer("A", adjacency()); self.idx = node_channel_indices()
        self.proj = nn.ModuleList([nn.Linear(2 * len(ix), node_dim) for ix in self.idx])      # last-step + window-mean per node
        self.layers = nn.ModuleList([GATLayer(node_dim, heads, dropout) for _ in range(layers)]); self.out = nn.Linear(2 * node_dim, out_dim); self.last_attention = None

    def forward(self, x, valid):
        vm = valid[:, :, None].float(); mean = (x * vm).sum(1) / vm.sum(1).clamp(min=1); last = x[:, -1]
        h = torch.stack([self.proj[k](torch.cat([last[:, ix], mean[:, ix]], 1)) for k, ix in enumerate(self.idx)], 1)    # (B,5,D)
        att = None
        for l in self.layers: h, att = l(h, self.A)
        self.last_attention = att
        return self.out(torch.cat([h.mean(1), h.max(1).values], 1))
