"""MOGAM: a two-layer GAT over the object co-occurrence graph, fused with the
visual, title, description and duration features by a learned gate.

This is the configuration selected on validation in the paper (Section 5):
class-identity node features, two GAT layers, mean pooling over nodes, one
token per modality, gated fusion, softmax classifier. The transformer fusion
alternative is kept as an option because it was selected for the daily vs.
high-risk task.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from .coco import NUM_CLASSES

HIDDEN = 256
IN_DIM = {"visual": 1000, "title": 768, "desc": 768, "dur": 1}
ALL_MODS = ("graph", "visual", "title", "desc", "dur")


class GraphEncoder(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.conv1 = GATConv(NUM_CLASSES, hidden)
        self.conv2 = GATConv(hidden, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, data):
        h = self.conv1(data.x, data.edge_index).relu()
        h = self.conv2(h, data.edge_index)
        return self.proj(global_mean_pool(h, data.batch))


class MOGAM(nn.Module):
    def __init__(self, num_classes, mods=ALL_MODS, fusion="gate", hidden=HIDDEN):
        super().__init__()
        self.mods, self.fusion = list(mods), fusion
        if "graph" in self.mods:
            self.graph = GraphEncoder(hidden)
            self.graph_norm = nn.LayerNorm(hidden)
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(IN_DIM[m], hidden), nn.LayerNorm(hidden))
            for m in self.mods if m != "graph"})
        k = len(self.mods)
        if fusion == "gate":
            self.score = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(),
                                       nn.Linear(hidden // 2, 1))
        elif fusion == "transformer":
            self.type_emb = nn.Parameter(torch.zeros(k, hidden))
            self.encoder = nn.TransformerEncoderLayer(
                hidden, nhead=4, dim_feedforward=hidden * 2, dropout=0.1, batch_first=True)
        else:
            raise ValueError(fusion)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(0.5), nn.Linear(hidden // 2, num_classes))
        self.last_gate = None

    @staticmethod
    def raw(data, m):
        """L2-normalised direction for vectors, log1p for duration (Section 5.3)."""
        if m == "dur":
            return torch.log1p(data.duration.float().clamp(min=0)).view(-1, 1)
        x = {"visual": data.visual, "title": data.title, "desc": data.description}[m].float()
        return F.normalize(x, dim=1)

    def set_stats(self, stats):
        """Per-dimension mean/std computed on the training split only."""
        for m, (mu, sd) in stats.items():
            self.register_buffer(f"mu_{m}", mu)
            self.register_buffer(f"sd_{m}", sd)

    def tokens(self, data):
        toks = []
        for m in self.mods:
            if m == "graph":
                toks.append(self.graph_norm(self.graph(data)))
            else:
                x = (self.raw(data, m) - getattr(self, f"mu_{m}")) / getattr(self, f"sd_{m}")
                toks.append(self.proj[m](x))
        return torch.stack(toks, dim=1)                      # [B, k, H]

    def forward(self, data):
        t = self.tokens(data)
        if self.fusion == "gate":
            w = torch.softmax(self.score(torch.tanh(t)).squeeze(-1), dim=1)
            self.last_gate = w.detach()
            z = (w.unsqueeze(-1) * t).sum(1)
        else:
            z = self.encoder(t + self.type_emb).mean(1)
        return self.head(z)
