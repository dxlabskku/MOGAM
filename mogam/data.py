"""Load the released corpus.

Expected layout (available from the authors on request, see README):

    data/
      labels.csv              id,group,channel      group in {daily, high_risk, depression}
      graphs/<id>.npy         (80, 80) float32 co-occurrence matrix (Eq. 1)
      features/<id>.npz       visual (1000,), title (768,), description (768,), duration (1,)

`channel` is an anonymised channel identifier used only for the user-disjoint split.
"""
import csv
import os

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from .coco import NUM_CLASSES

TASKS = {
    "daily_vs_depression": {"daily": 0, "depression": 1},
    "daily_vs_highrisk":   {"daily": 0, "high_risk": 1},
    "three_class":         {"daily": 0, "high_risk": 1, "depression": 2},
}


def load_records(root, task):
    labels = TASKS[task]
    recs = []
    with open(os.path.join(root, "labels.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["group"] not in labels:
                continue
            gp = os.path.join(root, "graphs", row["id"] + ".npy")
            fp = os.path.join(root, "features", row["id"] + ".npz")
            if not (os.path.exists(gp) and os.path.exists(fp)):
                continue
            recs.append({"id": row["id"], "label": labels[row["group"]],
                         "channel": row["channel"], "graph": gp, "feat": fp})
    if not recs:
        raise FileNotFoundError(f"no records for task {task} under {root}")
    return recs


def to_data(rec):
    m = np.load(rec["graph"]).astype(np.float32)
    assert m.shape == (NUM_CLASSES, NUM_CLASSES)
    A = nx.adjacency_matrix(nx.from_numpy_array(m))
    ei, ea = torch_geometric.utils.from_scipy_sparse_matrix(A)
    f = np.load(rec["feat"])
    d = Data(edge_index=ei, edge_attr=ea,
             x=torch.eye(NUM_CLASSES, dtype=torch.float32),      # class identity
             y=torch.LongTensor([rec["label"]]))
    d.visual = torch.tensor(f["visual"], dtype=torch.float32).view(1, -1)
    d.title = torch.tensor(f["title"], dtype=torch.float32).view(1, -1)
    d.description = torch.tensor(f["description"], dtype=torch.float32).view(1, -1)
    d.duration = torch.tensor(f["duration"], dtype=torch.float32).view(1, -1)
    return d


def split_indices(recs, seed, user_disjoint=False):
    """80/10/10 split redrawn per seed; optionally channel-disjoint."""
    n = len(recs)
    if user_disjoint:
        from sklearn.model_selection import GroupShuffleSplit
        idx = np.arange(n)
        groups = np.array([r["channel"] for r in recs])
        y = np.array([r["label"] for r in recs])
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(idx, y, groups))
        tr2, va = next(GroupShuffleSplit(1, test_size=0.125, random_state=seed).split(tr, y[tr], groups[tr]))
        return tr[tr2], tr[va], te
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(n * 0.8), int(n * 0.9)
    return idx[:a], idx[a:b], idx[b:]
